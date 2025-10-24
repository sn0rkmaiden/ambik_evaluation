import openai
from openai import OpenAI
import os, re, json
from utils.log import llm_log
import time
import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

import pickle
from collections import OrderedDict
from transformer_lens import HookedTransformer
from sae_lens import SAE

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
    HookedTransformer + SAE wrapper for gemma-2b-it that follows the repository LLM interface.
    - If `model` or `sae` are provided, they are used directly. Otherwise defaults try to load:
        SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
        HookedTransformer.from_pretrained(model_name, device=device)
    - request(prompt, stop=None, **kwargs) returns (output_text, message)
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
        max_new_tokens: int = 50,
    ) -> None:

        super().__init__(cache)
        self.model_name = model_name
        self.is_chat_version = True
        self.max_new_tokens = max_new_tokens
        
        print("HookedGEMMA init!")

        # device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Use provided sae/model if present, else load them
        if sae is not None:
            self.sae = sae
        else:
            # load SAE - this matches your snippet
            self.sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)

        if model is not None:
            self.model = model
        else:
            # load HookedTransformer model
            # note: pass device through to from_pretrained
            self.model = HookedTransformer.from_pretrained(model_name, device=device)

        # The HookedTransformer exposes tokenization helpers (to_tokens, to_string)
        # If you have a separate tokenizer object available in your environment, you can attach it as self.tokenizer.

    def _extract_assistant_answer(self, text: str) -> str:
        # Remove an echoed prompt and special tokens, returning the assistant reply.
        if text is None:    
            return ""
        # If the model echo contains "Assistant:", keep everything after the last occurrence
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1]
        # strip common special tokens and role names
        for token in ["<bos>", "<eos>", "System:", "User:"]:
            text = text.replace(token, "")
        return text.strip()

    def request(self, prompt, stop=None, **kwargs):
        """
        Accepts prompt (string or list of chat messages if you prefer).
        Returns (output_text, message) where message is a list of dicts matching repo expectation.
        """

        # Build message structure similar to other LLM classes in repo
        message = [{"role": "user", "content": prompt}]

        # support chaining previous_message like the repo's other LLMs
        if "previous_message" in kwargs and kwargs["previous_message"]:
            try:
                pm = kwargs["previous_message"]
                if isinstance(pm, list):
                    message = pm + message
                else:
                    # preserve previous_message if it's some other structure
                    message = kwargs["previous_message"]
            except Exception:
                message = kwargs["previous_message"]

        # Check cache
        cached = self.from_cache(message)
        if cached:
            message.append({"role": "assistant", "content": cached})
            return cached, message

        # prompt could be a list (chat) or a string; keep it as string for HookedTransformer
        prompt_text = prompt
        if isinstance(prompt, list):
            # If user passed chat-structured list, join into text (simple join - adapt if you have template)
            prompt_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in prompt])

        # Decide whether to prepend BOS (respect sae metadata)
        prepend_bos = bool(getattr(self.sae.cfg.metadata, "prepend_bos", False))

        # Tokenize / prepare input tokens for HookedTransformer
        # model.to_tokens accepts prepend_bos flag in your setup
        try:
            input_ids = self.model.to_tokens(prompt_text, prepend_bos=prepend_bos)
        except Exception:
            # fallback: try to call tokenizer if attached (rare for HookedTransformer)
            if hasattr(self, "tokenizer"):
                input_ids = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)["input_ids"]
            else:
                raise

        # Generate (no-steering path)
        with torch.no_grad():
            gen_out = self.model.generate(input_ids, max_new_tokens=self.max_new_tokens)
            # Convert tokens -> string. HookedTransformer usually provides to_string()
            if hasattr(self.model, "to_string"):
                decoded = self.model.to_string(gen_out)
            elif hasattr(self.model, "to_str_tokens"):
                decoded = self.model.to_str_tokens(gen_out)
            else:
                # Best-effort fallback
                decoded = str(gen_out)

        # decoded may be full text including prompt echo; extract assistant answer
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0]
        assistant_text = self._extract_assistant_answer(decoded)

        # log and cache
        super().log(prompt_text, assistant_text, model=self.model_name)
        self.save_to_cache(message, assistant_text)

        message.append({"role": "assistant", "content": assistant_text})
        return assistant_text, message

import os
import time
import json
from langchain_nebius import ChatNebius
from openai import OpenAI  # or whichever OpenAI‐wrapper you use

class CustomLLM(LLM):
    def __init__(self, name, api_key=None, cache=None) -> None:
        super().__init__(cache)
        name = name.strip().lower()
        # set api_key early
        self.api_key = api_key if api_key is not None else os.getenv("HF_TOKEN")
        # print(f"api_key is {self.api_key}, model name is {name}")

        # route depending on name
        if name == "deepseek":
            self.model_name = "deepseek-ai/DeepSeek-V3:nebius"
            # Use OpenAI client (via base_url) for this provider
            self.client_type = "openai"
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=self.api_key
            )

        elif name == "qwen":
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
        import torch

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
