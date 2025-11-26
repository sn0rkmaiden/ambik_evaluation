import openai
from openai import OpenAI
import os, re, json
from utils.log import llm_log
import time
import torch
import pickle
from collections import OrderedDict
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sae_lens import SAE
from langchain_nebius import ChatNebius
from gemma_adapter import generate_with_steering

class LLM:
    def __init__(self, cache = None) -> None:
        self.cache_path = cache
        self.cache = None
        self.cache_capacity = 100000
        if self.cache_path:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'rb') as file:
                    self.cache = pickle.load(file)
            else:
                self.cache = OrderedDict()


    def extract_json_string(self, input_string):
        def process_colons_string(input_string, colon_positions):
            def find_string_bounds(s, start_pos, next_pos=None):
                colon_pos = s.find(':', start_pos)
                quote_start = colon_pos + 1
                
                while quote_start < len(s) and s[quote_start] in ' \t\n':
                    quote_start += 1
                
                if s[quote_start] != '"':
                    return s
                
                string_start = quote_start
                
                if next_pos:
                    substring = s[string_start:next_pos]
                    quote_end = next_pos
                    escaped = 0
                    while quote_end > string_start:
                        quote_end -= 1
                        if s[quote_end] == '"' and escaped != 2:
                            escaped += 1
                        elif s[quote_end] == '"' and escaped == 2:
                            break
                else:
                    substring = s[string_start:]
                    quote_end = len(s)
                    while quote_end > string_start:
                        quote_end -= 1
                        if s[quote_end] == '"':
                            break
                        
                actual_string = s[string_start+1:quote_end]
                
                escaped_string = ""
                is_previous_backslash = False
            
                for char in actual_string:
                    if char == '"' and not is_previous_backslash:
                        escaped_string += '\\"'
                    else:
                        escaped_string += char
                    is_previous_backslash = (char == '\\')

                new_string = s[:string_start+1] + escaped_string + s[quote_end:]
                
                return new_string
            
            result_string = input_string
            length_change = 0
            for i, pos in enumerate(colon_positions):
                end_pos = colon_positions[i+1] if i+1 != len(colon_positions) else None
                if end_pos:
                    result_string = find_string_bounds(result_string, pos + length_change, end_pos+ length_change)
                else:
                    result_string = find_string_bounds(result_string, pos + length_change)
                length_change = len(result_string) - len(input_string)
            return result_string

        def find_colons(input_string):
            pattern = r'"\s*:\s*'
            matches = []
            for match in re.finditer(pattern, input_string):
                colon_pos = match.start() + match.group().find(':')
                matches.append(colon_pos)
            return matches
        
        def clean_json_string(json_str):
            json_str = json_str.replace("'", '"')
            json_str = re.sub(r'\bTrue\b', 'true', json_str)
            json_str = re.sub(r'\bFalse\b', 'false', json_str)  
            colon_positions = find_colons(json_str)
            json_str = process_colons_string(json_str, colon_positions)
            return json_str
        
        start = -1
        brace_count = 0
        
        for i, char in enumerate(input_string):
            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    json_str = input_string[start:i+1]
                    json_str = clean_json_string(json_str)
                    
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError as e:
                        print(json_str)
                        pass
                    start = -1

        return ''

    def from_cache(self, message):
        if self.cache:
            message_str = str(message)
            if message_str in self.cache:
                refresh = self.cache[message_str]
                self.cache.pop(message_str)
                self.cache[message_str] = refresh

                # Extract data from the existing cache.
                # if os.path.exists(self.cache_path+'.back'):
                #     with open(self.cache_path+'.back', 'rb') as file:
                #         self.cache_back = pickle.load(file)
                # else:
                #     self.cache_back = OrderedDict()
                # if message_str not in self.cache_back:
                #     self.cache_back[message_str] = refresh
                #     with open(self.cache_path+'.back', 'wb') as file:
                #         pickle.dump(self.cache_back, file)

                return refresh
        return None
    
    def save_to_cache(self, message, response):
        if self.cache_path:
            message_str = str(message)
            self.cache[message_str] = response
            with open(self.cache_path, 'wb') as file:
                pickle.dump(self.cache, file)
            if len(self.cache) > self.cache_capacity:
                self.cache.popitem(last=False)


    def request(self, prompt, stop, **kwargs):
        return
    
    def log(self, input, output, **kwargs):
        llm_log(input, output, **kwargs)
        pass

class HookedGEMMA(LLM):
    """
    HookedTransformer + SAE wrapper for Gemma that follows this repo's LLM interface.

    Features:
    - Optional SAE steering on a single feature (index).
    - Optional per-prompt max activation estimation (no ActivationStore required).
    - Falls back cleanly to unsteered generation when no feature is provided.

    Methods:
    - request(prompt, stop=None, **kwargs) -> (output_text, messages)
      * respects cache via self.from_cache / self.save_to_cache
      * accepts kwargs["previous_message"] (list of role-content dicts) like other models
      * accepts kwargs["json_format"] to force "JSON-only" response
    """

    def __init__(
        self,
        model_name: str = "gemma-2b-it",
        sae_release: str = "gemma-2b-it-res-jb",
        sae_id: str = "blocks.12.hook_resid_post",
        model=None,
        sae=None,
        cache: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 100,
        # steering options
        steering_feature: int | None = None,
        steering_strength: float = 1.0,
        max_act: float | None = None,           # fixed max act (if known)
        compute_max_per_turn: bool = False,     # estimate per prompt if True
    ) -> None:
        super().__init__(cache)

        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.is_chat_version = True  # evaluation code uses chat-style messages

        # device selection
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # load components if not injected
        self.model = model or HookedTransformer.from_pretrained(model_name, device=self.device, dtype=torch.float16)
        self.sae = sae or SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=self.device)

        # steering config
        self.steering_feature = steering_feature
        self.steering_strength = float(steering_strength)
        self.max_act = max_act
        self.compute_max_per_turn = bool(compute_max_per_turn)

    # ------------------------- helpers -------------------------

    def _build_text_from_messages(self, messages: list[dict], force_json: bool) -> str:
        """
        Convert OpenAI-style messages into a plain text conversation the Gemma adapter expects.
        """
        parts = []
        for m in messages:
            role = m.get("role", "").strip().lower()
            content = m.get("content", "").strip()
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                # fallback
                parts.append(f"{role.capitalize()}: {content}")

        # model should speak next:
        parts.append("Assistant:")
        text = "\n".join(parts)
        if force_json:
            text += "\nReturn only a valid JSON object without any extra text."
        return text

    def _extract_assistant_answer(self, decoded: str, prompt_text: str) -> str:
        """
        Strip the prompt echo if present; else take text after the last 'Assistant:'.
        """
        if decoded.startswith(prompt_text):
            # plain suffix
            return decoded[len(prompt_text):].strip()

        # else, split by the last "Assistant:" marker
        if "Assistant:" in decoded:
            return decoded.split("Assistant:")[-1].strip()
        return decoded.strip()

    def _estimate_feature_max_for_prompt(self, prompt_text: str, feature_idx: int) -> float:
        """
        Compute a quick per-prompt max activation for `feature_idx` by running to the SAE hook
        and encoding into feature space. No ActivationStore needed.
        """
        hook_name = self.sae.cfg.metadata.hook_name
        try:
            prepend_bos = bool(getattr(self.sae.cfg.metadata, "prepend_bos", False))
        except Exception:
            prepend_bos = False

        toks = self.model.to_tokens(prompt_text, prepend_bos=prepend_bos)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(toks, names_filter=[hook_name])
            sae_in = cache[hook_name]
            feats = self.sae.encode(sae_in)              # [batch, seq, n_feat]
            feats = feats.reshape(-1, feats.shape[-1])   # flatten batch+seq
            max_val = feats[:, feature_idx].max().item()
        return float(max_val if max_val != float("inf") else 1.0)

    # ------------------------- main API -------------------------

    def request(self, prompt, stop=None, **kwargs):
        """Return (assistant_text, messages) — mirrors other LLMs here."""
        # Build messages list (for cache compatibility)
        if "previous_message" in kwargs and isinstance(kwargs["previous_message"], list):
            messages = kwargs["previous_message"] + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        # cache hit?
        cached = self.from_cache(messages)
        if cached:
            messages.append({"role": "assistant", "content": cached})
            return cached, messages

        # Build text prompt expected by our steered generator
        force_json = bool(kwargs.get("json_format", False))
        prompt_text = self._build_text_from_messages(messages, force_json)

        # Decide BOS policy from SAE metadata
        prepend_bos = bool(getattr(self.sae.cfg.metadata, "prepend_bos", False))

        # Choose steered vs unsteered
        do_steer = self.steering_feature is not None

        if do_steer:
            # pick max_act (fixed or estimated)
            turn_max = self.max_act
            if self.compute_max_per_turn or turn_max is None:
                try:
                    turn_max = self._estimate_feature_max_for_prompt(prompt_text, int(self.steering_feature))
                except Exception:
                    turn_max = 1.0

            # run steered generation
            try:
                # Ensure adapter and llm agree on BOS policy for tokenization
                original_prepend = getattr(self.sae.cfg.metadata, "prepend_bos", False)
                self.sae.cfg.metadata.prepend_bos = prepend_bos

                decoded = generate_with_steering(
                    model=self.model,
                    sae=self.sae,
                    prompt=prompt_text,
                    steering_feature=int(self.steering_feature),
                    max_act=float(turn_max),
                    steering_strength=float(self.steering_strength),
                    max_new_tokens=self.max_new_tokens,
                )
            finally:
                self.sae.cfg.metadata.prepend_bos = original_prepend
        else:
            # unsteered path
            with torch.no_grad():
                toks = self.model.to_tokens(prompt_text, prepend_bos=prepend_bos)
                gen_out = self.model.generate(
                    toks,
                    max_new_tokens=self.max_new_tokens,
                    stop_at_eos=False if self.model.cfg.device == "mps" else True,
                    prepend_bos=prepend_bos,
                )
                # HookedTransformer exposes tokenizer.decode
                decoded = self.model.tokenizer.decode(gen_out[0])

        # Post-process: only keep assistant answer (no prompt echo)
        assistant_text = self._extract_assistant_answer(decoded, prompt_text)

        # log + cache + return
        super().log(prompt_text, assistant_text, model=self.model_name)
        self.save_to_cache(messages, assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text, messages

class CustomLLM(LLM):
    def __init__(self, name, api_key=None, cache=None) -> None:
        super().__init__(cache)
        name = name.strip().lower()
        
        # print(f"api_key is {self.api_key}, model name is {name}")
        # route depending on name
        if name == "deepseek":
            self.api_key = api_key if api_key is not None else os.getenv("HF_TOKEN")
            self.model_name = "deepseek-ai/DeepSeek-V3:nebius"
            # Use OpenAI client (via base_url) for this provider
            self.client_type = "openai"
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.api_key
            )

        elif name == "qwen":
            self.api_key = api_key if api_key is not None else os.getenv("NEBIUS_API_KEY")
            self.model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"
            # Choose to use ChatNebius for this provider
            self.client_type = "nebius"
            self.client = ChatNebius(
                model=self.model_name,
                temperature=0.00001,      # or whatever default you like
                api_key=self.api_key,
                base_url="https://api.studio.nebius.com/v1/"
            )

        else:
            raise ValueError(f"Unknown model name: {name}")

        print(f"CustomLLM {self.model_name} init! (client_type={self.client_type})")

    def request(self, prompt, stop=None, **kwargs):
        # build message list
        message = [{"role": "user", "content": prompt}]

        prev = kwargs.get("previous_message")
        if prev:
            message = list(prev) + message

        json_format = bool(kwargs.get("json_format", False))

        # if nebius + json_format => prepend JSON instruction to prompt
        if self.client_type == "nebius" and json_format:
            instruction = (
                "\nReturn **only** a valid JSON object without any extra text."
            )
            message[0]["content"] = instruction + "\n" + message[0]["content"]

        # caching check (assuming your base class implements from_cache)
        response = self.from_cache(message)
        if response:
            message.append({"role": "assistant", "content": response})
            return response, message

        # depending on client type, call differently
        if self.client_type == "openai":
            # original path
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    messages=message,
                    stop=stop,
                    **({"response_format": {"type": "json_object"}} if json_format else {})
                )
                time.sleep(0.5)
            except Exception as e:
                time.sleep(5)
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    messages=message,
                    stop=stop,
                    **({"response_format": {"type": "json_object"}} if json_format else {})
                )
                time.sleep(0.5)

            output_text = completion.choices[0].message.content
            usage = getattr(completion, "usage", None)

        elif self.client_type == "nebius":
            # ChatNebius path
            try:
                response_obj = self.client.invoke(message)  # doc: ChatNebius.invoke returns an object with .content
                output_text = response_obj.content
                usage = getattr(response_obj, "usage", None)  # if available
            except Exception as e:
                # maybe retry once
                time.sleep(5)
                response_obj = self.client.invoke(message)
                output_text = response_obj.content
                usage = getattr(response_obj, "usage", None)

        else:
            raise RuntimeError(f"Unknown client_type: {self.client_type}")

        # parse usage metrics if present
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
        else:
            prompt_tokens = completion_tokens = total_tokens = None

        # logging (assuming base class log method)
        super().log(
            message,
            output_text,
            model=self.model_name,
            system_fingerprint=getattr(usage, "system_fingerprint", None),
            usage=[prompt_tokens, completion_tokens, total_tokens]
        )

        # caching
        self.save_to_cache(message, output_text)
        message.append({"role": "assistant", "content": output_text})
        return output_text, message

class HuggingFaceLLM(LLM):
    def __init__(self, name, cache=None):
        # Set deterministic behavior
        # torch.manual_seed(8848)
        # torch.cuda.manual_seed_all(8848)

        super().__init__(cache)

        model_id = name.strip()
        print(f"Huggingface model {model_id} init!")
        self.max_new_tokens = 100
        self.is_chat_version = "-chat" in model_id.lower()

        config = AutoConfig.from_pretrained(model_id)
        self.model_name = config.model_type

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            return_dict=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def request(self, prompt, stop=None, **kwargs):

        if self.is_chat_version:
            if 'previous_message' in kwargs:
                messages = kwargs['previous_message'] + [{"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        cached = self.from_cache(messages)
        if cached:
            messages.append({"role": "assistant", "content": cached})
            return cached, messages

        # --- Tokenize & move to GPU ---
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # print(f"[LLM REQUEST] Prompt length: {len(prompt)}")  
        # print(f"[LLM REQUEST] Last 200 chars of prompt: {prompt[-200:]}") 

        # --- Generate text ---
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=1.0,
                temperature=1.0,
                min_length=None,
                use_cache=True,
                top_k=50,
                repetition_penalty=1.0,
                length_penalty=1,
            )

            # Decode only the newly generated part
            output_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        # print(f"[LLM RESPONSE] Response: {output_text}")  

        # --- Log + cache ---
        super().log(prompt, output_text, model=self.model_name)
        self.save_to_cache(messages, output_text)
        messages.append({"role": "assistant", "content": output_text})

        return output_text, messages
