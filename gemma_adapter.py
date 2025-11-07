import torch
from tqdm import tqdm
from functools import partial
import re


def find_max_activation(model, sae, activation_store, feature_idx, num_batches=100):
    """
    Find the maximum activation for a given feature index. This is useful for
    calibrating the right amount of the feature to add.
    """
    max_activation = 0.0

    pbar = tqdm(range(num_batches))
    for _ in pbar:
        tokens = activation_store.get_batch_tokens()

        layer = int(re.search(r"\.(\d+)\.", sae.cfg.metadata.hook_name).group(1))  # type: ignore
        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=layer + 1,
            names_filter=[sae.cfg.metadata.hook_name],
        )
        sae_in = cache[sae.cfg.metadata.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        batch_max_activation = feature_acts[:, feature_idx].max().item()
        max_activation = max(max_activation, batch_max_activation)

        pbar.set_description(f"Max activation: {max_activation:.4f}")

    return max_activation


def steering(
    activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0
):
    # Note if the feature fires anyway, we'd be adding to that here.
    return activations + max_act * steering_strength * steering_vector


def generate_with_steering(
    model,
    sae,
    prompt,
    steering_feature,
    max_act,
    steering_strength=1.0,
    max_new_tokens=95,
):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.metadata.prepend_bos)

    steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)

    steering_hook = partial(
        steering,
        steering_vector=steering_vector,
        steering_strength=steering_strength,
        max_act=max_act,
    )

    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[(sae.cfg.metadata.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=False if model.cfg.device == "mps" else True,
            prepend_bos=sae.cfg.metadata.prepend_bos,
        )

    return model.tokenizer.decode(output[0])

def run_multiturn_conversation_generic(
    model,
    sae,
    activation_store,
    steering_feature=None,          # int or None
    steering_strength=1.0,
    max_batches_for_max=30,
    compute_max_per_turn=False,
    instruction_tuned=True,         # True for gemma-2b-it-like models
    use_token_append=True,          # True => keep token tensor and append new tokens (recommended)
    max_new_tokens=128,
    system_prompt="System: You are a helpful assistant.",
    max_history_turns=6,
    num_turns=10,
):

    # Decide prepend_bos policy (default: use sae metadata)
    prepend_bos_default = bool(getattr(sae.cfg.metadata, "prepend_bos", False))

    # Calibrate max_act once unless user requests per-turn recalculation
    max_act = None
    if not compute_max_per_turn and steering_feature is not None:
        print(f"[setup] estimating max activation (batches={max_batches_for_max}) for feature {steering_feature} ...")
        max_act = find_max_activation(model, sae, activation_store, steering_feature, num_batches=max_batches_for_max)
        print(f"[setup] max_act ≈ {max_act:.4f}")

    # Conversation history: list of (user, assistant) pairs (assistant may be empty initially)
    history = []

    # Token append state
    token_buffer = None
    bos_added = False

    def build_prompt_text(history, current_user):
        # keep last N turns
        hist = history[-max_history_turns:]
        parts = [system_prompt.strip()]
        for u, a in hist:
            parts.append(f"\nUser: {u}")
            if a:
                parts.append(f"\nAssistant: {a}")
        parts.append(f"\nUser: {current_user}\nAssistant:")
        return "".join(parts)

    def extract_assistant_answer(text: str) -> str:
        # Get portion after last Assistant:
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1]
        # Remove special tokens and whitespace
        for token in ["<bos>", "<eos>", "System:", "User:"]:
            text = text.replace(token, "")
        return text.strip()

    for turn in range(num_turns):
        user_text = input(f"User (turn {turn+1}/{num_turns}): ").strip()
        if not user_text:
            print("[info] empty input; ending conversation.")
            break

        # Recompute max_act per-turn if requested
        if compute_max_per_turn and steering_feature is not None:
            max_act = find_max_activation(model, sae, activation_store, steering_feature, num_batches=max_batches_for_max)
            print(f"[info] computed per-turn max_act = {max_act:.4f}")

        # Build prompt text (formatted for instruction/chat models)
        prompt_text = build_prompt_text(history, user_text)

        # Determine effective prepend_bos for this generation (only once if configured)
        effective_prepend_bos = prepend_bos_default and (not bos_added)

        # If we are keeping tokens and already have token_buffer, append user tokens without BOS
        if use_token_append and token_buffer is not None:
            # Append the user message tokens (no BOS)
            user_tokens = model.to_tokens(f" User: {user_text}\nAssistant:", prepend_bos=False)
            token_buffer = torch.cat([token_buffer, user_tokens], dim=-1)
            input_for_gen = token_buffer
        else:
            # Re-tokenize full prompt; pass prepend_bos for first generation only
            input_for_gen = model.to_tokens(prompt_text, prepend_bos=effective_prepend_bos)

        if steering_feature is not None:
            # Use your SAE steering helper that expects (model, sae, prompt_text, ..)
            # generate_with_steering in your notebook uses sae.cfg.metadata.prepend_bos internally,
            # so to be safe, temporarily set it to effective_prepend_bos:
            original_prepend = getattr(sae.cfg.metadata, "prepend_bos", False)
            sae.cfg.metadata.prepend_bos = effective_prepend_bos
            assistant_text = generate_with_steering(
                model, sae, prompt_text, steering_feature, max_act, steering_strength=steering_strength, max_new_tokens=max_new_tokens
            )

            assistant_text = extract_assistant_answer(assistant_text)
            
            sae.cfg.metadata.prepend_bos = original_prepend
        else:
            # Plain generation: model.generate expects token tensor
            gen_out = model.generate(input_for_gen, max_new_tokens=max_new_tokens)
            # convert tokens -> text. Try common APIs, adapt if your model uses different name.
            if hasattr(model, "to_string"):
                assistant_text = model.to_string(gen_out)
            elif hasattr(model, "to_str_tokens"):
                assistant_text = model.to_str_tokens(gen_out)
            else:
                # fallback: assume HF tokenizer is available on model.tokenizer
                try:
                    assistant_text = model.tokenizer.decode(gen_out[0].cpu().numpy(), skip_special_tokens=True)
                except Exception:
                    assistant_text = str(gen_out)

            assistant_text = assistant_text[0]
            assistant_text = extract_assistant_answer(assistant_text)

        # Print and update history
        print("\nAssistant:", assistant_text.strip(), "\n")
        history.append((user_text, assistant_text.strip()))

        # Update token_buffer: append assistant tokens (no BOS) if using token append
        if use_token_append:
            if token_buffer is None:
                # create initial buffer from prompt_text; ensure BOS only if effective_prepend_bos
                token_buffer = model.to_tokens(prompt_text, prepend_bos=effective_prepend_bos)
            assistant_tokens = model.to_tokens(" " + assistant_text.strip(), prepend_bos=False)
            token_buffer = torch.cat([token_buffer, assistant_tokens], dim=-1)

        # mark that BOS has been used
        if effective_prepend_bos:
            bos_added = True