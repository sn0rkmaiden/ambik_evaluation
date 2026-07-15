#!/usr/bin/env python3
"""Matched baseline-vs-steered generation benchmark for Gemma + SAE vectors.

Run this script from ambik_evaluation-main. It bypasses the result cache and metric
models. Generation length is fixed, so response-length changes cannot bias latency.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def unwrap_sae(value):
    return value[0] if isinstance(value, tuple) else value


def get_hook_name(sae) -> str:
    if hasattr(sae.cfg, "hook_name"):
        return sae.cfg.hook_name
    md = getattr(sae.cfg, "metadata", None)
    if md is not None and hasattr(md, "hook_name"):
        return md.hook_name
    raise ValueError("Cannot infer hook_name from SAE config")


def prepend_bos_for(sae) -> bool:
    if hasattr(sae.cfg, "prepend_bos"):
        return bool(sae.cfg.prepend_bos)
    md = getattr(sae.cfg, "metadata", None)
    return bool(getattr(md, "prepend_bos", False))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def param_bytes(module) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def make_prompts(csv_path: str, prompt_template_path: str, n: int, seed: int) -> list[str]:
    df = pd.read_csv(csv_path).sample(n=n, random_state=seed).reset_index(drop=True)
    template = Path(prompt_template_path).read_text(encoding="utf-8")
    prompts = []
    for _, row in df.iterrows():
        p = template.replace("<DESCRIPTION>", str(row["environment_full"]))
        p = p.replace("<TASK>", str(row["ambiguous_task"]))
        prompts.append("User: " + p.strip() + "\nAssistant:")
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--sae_release", required=True)
    ap.add_argument("--sae_id", required=True)
    ap.add_argument("--feature", type=int, required=True)
    ap.add_argument("--strength", type=float, required=True)
    ap.add_argument("--max_act", type=float, default=4.0)
    ap.add_argument("--per_prompt_max", action="store_true")
    ap.add_argument("--dataset_csv", default="data/ambik_calib_100.csv")
    ap.add_argument("--prompt_template", default="data/prompt_draft.txt")
    ap.add_argument("--num_prompts", type=int, default=20)
    ap.add_argument("--new_tokens", type=int, default=32)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = make_prompts(args.dataset_csv, args.prompt_template, args.num_prompts, args.seed)

    sync(); t0 = time.perf_counter()
    model = HookedTransformer.from_pretrained(
        args.model_name, device=device, dtype=torch.float16
    )
    sync(); model_load_seconds = time.perf_counter() - t0

    sync(); t0 = time.perf_counter()
    sae = unwrap_sae(SAE.from_pretrained(
        release=args.sae_release, sae_id=args.sae_id, device=device
    ))
    sync(); sae_load_seconds = time.perf_counter() - t0

    hook_name = get_hook_name(sae)
    prepend_bos = prepend_bos_for(sae)
    sae_resident_bytes = param_bytes(sae)
    allocated_with_sae = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None

    steering_vector = sae.W_dec[int(args.feature)].detach().to(device).clone()
    vector_bytes = steering_vector.numel() * steering_vector.element_size()

    def estimate_prompt_max(tokens: torch.Tensor) -> tuple[float, float]:
        sync(); t = time.perf_counter()
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = sae.encode(cache[hook_name]).reshape(-1, sae.cfg.d_sae)
            value = float(acts[:, int(args.feature)].max().item())
        sync(); elapsed = time.perf_counter() - t
        return value, elapsed

    # For fixed-scale deployment the full SAE is unnecessary after extracting W_dec.
    if not args.per_prompt_max:
        sae.to("cpu")
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    allocated_vector_only = int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None

    tokenized = [model.to_tokens(p, prepend_bos=prepend_bos).to(device) for p in prompts]

    def generate(tokens: torch.Tensor, steered: bool) -> tuple[float, float]:
        calibration_seconds = 0.0
        local_max = float(args.max_act)
        if steered and args.per_prompt_max:
            local_max, calibration_seconds = estimate_prompt_max(tokens)

        def hook_fn(value, hook):
            return value + (local_max * float(args.strength)) * steering_vector

        sync(); t = time.perf_counter()
        with torch.inference_mode():
            if steered:
                with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                    model.generate(
                        tokens,
                        max_new_tokens=args.new_tokens,
                        stop_at_eos=False,
                        do_sample=False,
                        use_past_kv_cache=True,
                        prepend_bos=prepend_bos,
                        return_type="tokens",
                        verbose=False,
                    )
            else:
                model.generate(
                    tokens,
                    max_new_tokens=args.new_tokens,
                    stop_at_eos=False,
                    do_sample=False,
                    use_past_kv_cache=True,
                    prepend_bos=prepend_bos,
                    return_type="tokens",
                    verbose=False,
                )
        sync(); generation_seconds = time.perf_counter() - t
        return generation_seconds, calibration_seconds

    # Warm both paths to exclude compilation/cache setup.
    for i in range(args.warmups):
        generate(tokenized[i % len(tokenized)], False)
        generate(tokenized[i % len(tokenized)], True)

    def run_condition(steered: bool) -> dict[str, Any]:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        latencies: list[float] = []
        calibration: list[float] = []
        for toks in tokenized:
            gen_s, cal_s = generate(toks, steered)
            latencies.append(gen_s)
            calibration.append(cal_s)
        return {
            "generation_seconds_per_prompt": latencies,
            "calibration_seconds_per_prompt": calibration,
            "median_generation_seconds": statistics.median(latencies),
            "p95_generation_seconds": percentile(latencies, 0.95),
            "median_ms_per_generated_token": statistics.median(latencies) * 1000.0 / args.new_tokens,
            "p95_ms_per_generated_token": percentile(latencies, 0.95) * 1000.0 / args.new_tokens,
            "median_generated_tokens_per_second": args.new_tokens / statistics.median(latencies),
            "median_calibration_seconds": statistics.median(calibration),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        }

    baseline = run_condition(False)
    steered = run_condition(True)
    overhead_pct = 100.0 * (
        steered["median_generation_seconds"] / baseline["median_generation_seconds"] - 1.0
    )
    end_to_end_steered = steered["median_generation_seconds"] + steered["median_calibration_seconds"]
    end_to_end_overhead_pct = 100.0 * (
        end_to_end_steered / baseline["median_generation_seconds"] - 1.0
    )

    gpu = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": torch.cuda.get_device_name(),
            "total_memory_bytes": int(p.total_memory),
            "compute_capability": f"{p.major}.{p.minor}",
            "cuda_runtime": torch.version.cuda,
        }

    result = {
        "kind": "online_latency_profile",
        "arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": gpu,
        },
        "model_load_seconds": model_load_seconds,
        "sae_load_seconds": sae_load_seconds,
        "hook_name": hook_name,
        "memory_accounting": {
            "sae_parameter_and_buffer_bytes": sae_resident_bytes,
            "one_decoder_vector_bytes": int(vector_bytes),
            "allocated_with_full_sae_bytes": allocated_with_sae,
            "allocated_after_vector_only_conversion_bytes": allocated_vector_only,
        },
        "baseline": baseline,
        "steered": steered,
        "median_generation_latency_overhead_percent": overhead_pct,
        "median_end_to_end_overhead_including_per_prompt_calibration_percent": end_to_end_overhead_pct,
        "prompt_token_lengths": [int(x.shape[-1]) for x in tokenized],
    }
    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved profile: {out}")


if __name__ == "__main__":
    main()
