#!/usr/bin/env python3
"""
Clear-input AmbiK evaluation for the reported Gemma-9B configuration.

This script is intentionally standalone with respect to the repository's
llm.py and gemma_adapter.py. Those files assume sae.cfg.metadata, which is
incompatible with some SAE Lens versions/config formats.

Run from the root of the ambik_evaluation repository:

    CUDA_VISIBLE_DEVICES=0 python run_clear_ambik_gemma9b_standalone.py \
        --num-examples 2

After the two-example smoke test succeeds:

    CUDA_VISIBLE_DEVICES=0 python run_clear_ambik_gemma9b_standalone.py \
        --num-examples 100

The script reproduces the relevant settings recorded in the successful run:
    model: gemma-2-9b-it
    SAE release: gemma-scope-9b-it-res-canonical
    SAE id: layer_20/width_16k/canonical
    hook: blocks.20.hook_resid_post
    feature: 166
    strength: 30.0
    max_act: 4.0
    max_new_tokens: 100
    dataset sampling seed: 50

It evaluates the paired `unambiguous_direct` instructions and reports:
    - over-asking rate: at least one generated question
    - predicted-ambiguous false-positive rate
    - Wilson 95% confidence intervals
    - paired bootstrap CI for the change
    - exact two-sided McNemar p-value
"""

from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from utils.parsing import parse_model_json


Z_975 = 1.959963984540054


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_sae(loaded: Any) -> Any:
    """Support SAE Lens versions returning either SAE or (SAE, cfg, sparsity)."""
    if isinstance(loaded, tuple):
        if not loaded:
            raise RuntimeError("SAE.from_pretrained() returned an empty tuple.")
        return loaded[0]
    return loaded


def config_value(sae: Any, name: str, default: Any = None) -> Any:
    """
    Read a setting from either:
      - newer layout: sae.cfg.metadata.<name>
      - older layout: sae.cfg.<name>
    """
    cfg = getattr(sae, "cfg", None)
    if cfg is None:
        return default

    metadata = getattr(cfg, "metadata", None)
    if metadata is not None:
        value = getattr(metadata, name, None)
        if value is not None:
            return value

    value = getattr(cfg, name, None)
    return default if value is None else value


def build_dataset_prompt(environment: str, task: str, template: str) -> str:
    return template.replace("<DESCRIPTION>", str(environment)).replace("<TASK>", str(task))


def build_request_prompt(dataset_prompt: str) -> str:
    """
    Match the original runner.py -> get_model_questions() -> HookedGEMMA.request()
    prompt construction.

    get_model_questions() appends one JSON-only instruction, and request() adds
    the User/Assistant wrapper plus another JSON-only instruction.
    """
    instruction = (
        dataset_prompt
        + "\nReturn **only** a valid JSON object without any extra text."
    )
    return (
        f"User: {instruction}\n"
        "Assistant:\n"
        "Return only a valid JSON object without any extra text."
    )


def extract_assistant_answer(decoded: str, prompt_text: str) -> str:
    """Match HookedGEMMA._extract_assistant_answer()."""
    if decoded.startswith(prompt_text):
        return decoded[len(prompt_text):].strip()
    if "Assistant:" in decoded:
        return decoded.split("Assistant:")[-1].strip()
    return decoded.strip()


def steering_hook(
    activations: torch.Tensor,
    hook: Any,
    *,
    steering_vector: torch.Tensor,
    steering_strength: float,
    max_act: float,
) -> torch.Tensor:
    del hook
    return activations + max_act * steering_strength * steering_vector


@torch.no_grad()
def generate_baseline(
    model: HookedTransformer,
    prompt_text: str,
    *,
    prepend_bos: bool,
    max_new_tokens: int,
) -> str:
    """
    Match the unsteered branch of HookedGEMMA.request():
    no explicit temperature or top_p arguments.
    """
    tokens = model.to_tokens(prompt_text, prepend_bos=prepend_bos)
    output = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        stop_at_eos=False if model.cfg.device == "mps" else True,
        prepend_bos=prepend_bos,
    )
    decoded = model.tokenizer.decode(output[0])
    return extract_assistant_answer(decoded, prompt_text)


@torch.no_grad()
def generate_steered(
    model: HookedTransformer,
    sae: Any,
    prompt_text: str,
    *,
    hook_name: str,
    feature: int,
    strength: float,
    max_act: float,
    prepend_bos: bool,
    max_new_tokens: int,
) -> str:
    """
    Match gemma_adapter.generate_with_steering() from the repository.
    """
    tokens = model.to_tokens(prompt_text, prepend_bos=prepend_bos)
    steering_vector = sae.W_dec[int(feature)].to(model.cfg.device)

    hook_fn = partial(
        steering_hook,
        steering_vector=steering_vector,
        steering_strength=float(strength),
        max_act=float(max_act),
    )

    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=False if model.cfg.device == "mps" else True,
            prepend_bos=prepend_bos,
        )

    decoded = model.tokenizer.decode(output[0])
    return extract_assistant_answer(decoded, prompt_text)


def normalize_questions(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def evaluate_condition(
    *,
    condition: str,
    model: HookedTransformer,
    sae: Any,
    rows: pd.DataFrame,
    prompt_template: str,
    hook_name: str,
    feature: int,
    strength: float,
    max_act: float,
    prepend_bos: bool,
    max_new_tokens: int,
    generation_seed: int,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(rows.iterrows()):
        # Same per-example seed in the baseline and steered conditions.
        set_seed(generation_seed + position)

        dataset_prompt = build_dataset_prompt(
            row.get("environment_full", ""),
            row.get("unambiguous_direct", ""),
            prompt_template,
        )
        request_prompt = build_request_prompt(dataset_prompt)

        if condition == "baseline":
            raw = generate_baseline(
                model,
                request_prompt,
                prepend_bos=prepend_bos,
                max_new_tokens=max_new_tokens,
            )
        elif condition == "steered":
            raw = generate_steered(
                model,
                sae,
                request_prompt,
                hook_name=hook_name,
                feature=feature,
                strength=strength,
                max_act=max_act,
                prepend_bos=prepend_bos,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        parsed = parse_model_json(raw)
        if parsed is None:
            record = {
                "id": int(row.get("id", position)),
                "condition": condition,
                "instruction": str(row.get("unambiguous_direct", "")),
                "parse_error": True,
                "predicted_ambiguous": None,
                "model_questions": None,
                "asked_question": None,
                "num_questions": None,
                "raw_output": raw,
            }
            results.append(record)
            checkpoint_path.write_text(
                json.dumps({"examples": results}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            raise ValueError(
                f"Could not parse JSON for id={record['id']}. "
                f"The partial outputs were saved to {checkpoint_path}."
            )

        questions = normalize_questions(
            parsed.get("question", parsed.get("questions", []))
        )
        record = {
            "id": int(row.get("id", position)),
            "condition": condition,
            "instruction": str(row.get("unambiguous_direct", "")),
            "parse_error": False,
            "predicted_ambiguous": bool(parsed.get("ambiguous", False)),
            "model_questions": questions,
            "asked_question": len(questions) > 0,
            "num_questions": len(questions),
            "raw_output": raw,
        }
        results.append(record)

        # Save after every example so a failure does not lose completed outputs.
        checkpoint_path.write_text(
            json.dumps({"examples": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"[{condition}] {position + 1:03d}/{len(rows)} "
            f"id={record['id']} "
            f"ambiguous={record['predicted_ambiguous']} "
            f"questions={record['num_questions']}"
        )

    return results


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = successes / n
    z = Z_975
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(baseline_only: int, steered_only: int) -> float:
    discordant = baseline_only + steered_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, steered_only)
    lower_tail = (
        sum(math.comb(discordant, i) for i in range(smaller + 1))
        / (2.0 ** discordant)
    )
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_ci(
    baseline: np.ndarray,
    steered: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 20000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    differences = steered - baseline
    estimates = np.empty(n_bootstrap, dtype=float)

    for start in range(0, n_bootstrap, 2000):
        stop = min(start + 2000, n_bootstrap)
        indices = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = differences[indices].mean(axis=1)

    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_binary_pair(
    baseline: np.ndarray,
    steered: np.ndarray,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    n = len(baseline)
    baseline_count = int(baseline.sum())
    steered_count = int(steered.sum())
    baseline_only = int(np.sum((baseline == 1) & (steered == 0)))
    steered_only = int(np.sum((baseline == 0) & (steered == 1)))

    return {
        "n": n,
        "baseline_count": baseline_count,
        "baseline_rate": baseline_count / n,
        "baseline_wilson_95ci": list(wilson_interval(baseline_count, n)),
        "steered_count": steered_count,
        "steered_rate": steered_count / n,
        "steered_wilson_95ci": list(wilson_interval(steered_count, n)),
        "paired_change": float(steered.mean() - baseline.mean()),
        "paired_change_bootstrap_95ci": list(
            paired_bootstrap_ci(
                baseline,
                steered,
                seed=bootstrap_seed,
            )
        ),
        "baseline_only": baseline_only,
        "steered_only": steered_only,
        "exact_mcnemar_p": exact_mcnemar_p(
            baseline_only,
            steered_only,
        ),
    }


def summarize(
    baseline_results: list[dict[str, Any]],
    steered_results: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_by_id = {int(x["id"]): x for x in baseline_results}
    steered_by_id = {int(x["id"]): x for x in steered_results}
    ids = sorted(set(baseline_by_id) & set(steered_by_id))

    if not ids:
        raise ValueError("No paired example IDs were found.")

    for item_id in ids:
        if baseline_by_id[item_id]["parse_error"]:
            raise ValueError(f"Baseline parse error for id={item_id}.")
        if steered_by_id[item_id]["parse_error"]:
            raise ValueError(f"Steered parse error for id={item_id}.")

    baseline_asked = np.asarray(
        [baseline_by_id[i]["asked_question"] for i in ids],
        dtype=float,
    )
    steered_asked = np.asarray(
        [steered_by_id[i]["asked_question"] for i in ids],
        dtype=float,
    )
    baseline_ambiguous = np.asarray(
        [baseline_by_id[i]["predicted_ambiguous"] for i in ids],
        dtype=float,
    )
    steered_ambiguous = np.asarray(
        [steered_by_id[i]["predicted_ambiguous"] for i in ids],
        dtype=float,
    )

    return {
        "n_paired": len(ids),
        "overasking": summarize_binary_pair(
            baseline_asked,
            steered_asked,
            bootstrap_seed=bootstrap_seed,
        ),
        "predicted_ambiguous_false_positive": summarize_binary_pair(
            baseline_ambiguous,
            steered_ambiguous,
            bootstrap_seed=bootstrap_seed + 1,
        ),
    }


def print_binary_summary(title: str, result: dict[str, Any]) -> None:
    print(f"\n{title}")
    print(
        f"Baseline: {result['baseline_count']}/{result['n']} "
        f"= {result['baseline_rate']:.3f} "
        f"[{result['baseline_wilson_95ci'][0]:.3f}, "
        f"{result['baseline_wilson_95ci'][1]:.3f}]"
    )
    print(
        f"Steered:  {result['steered_count']}/{result['n']} "
        f"= {result['steered_rate']:.3f} "
        f"[{result['steered_wilson_95ci'][0]:.3f}, "
        f"{result['steered_wilson_95ci'][1]:.3f}]"
    )
    print(
        f"Change:   {result['paired_change']:+.3f} "
        f"[{result['paired_change_bootstrap_95ci'][0]:+.3f}, "
        f"{result['paired_change_bootstrap_95ci'][1]:+.3f}]"
    )
    print(
        "Discordant pairs: "
        f"baseline-only={result['baseline_only']}, "
        f"steered-only={result['steered_only']}"
    )
    print(f"Exact McNemar p = {result['exact_mcnemar_p']:.6g}")


def resolve_prepend_bos(sae: Any, model: HookedTransformer) -> bool:
    """
    Prefer the value stored with the SAE. Fall back to TransformerLens's model
    configuration only when the SAE does not expose it.
    """
    value = config_value(sae, "prepend_bos", None)
    if value is not None:
        return bool(value)

    model_value = getattr(model.cfg, "default_prepend_bos", None)
    if model_value is not None:
        return bool(model_value)

    raise RuntimeError(
        "Could not determine prepend_bos from either the SAE or model config. "
        "Run with --prepend-bos true or --prepend-bos false."
    )


def parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected one of: auto, true, false."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ambik_calib_100.csv")
    parser.add_argument("--prompt-template", default="data/prompt_draft.txt")
    parser.add_argument("--output-dir", default="results/clear_input_eval")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--dataset-seed", type=int, default=50)
    parser.add_argument("--generation-seed", type=int, default=50)

    parser.add_argument("--model", default="gemma-2-9b-it")
    parser.add_argument(
        "--sae-release",
        default="gemma-scope-9b-it-res-canonical",
    )
    parser.add_argument(
        "--sae-id",
        default="layer_20/width_16k/canonical",
    )
    parser.add_argument(
        "--hook-name",
        default="blocks.20.hook_resid_post",
    )
    parser.add_argument("--feature", type=int, default=166)
    parser.add_argument("--strength", type=float, default=30.0)
    parser.add_argument("--max-act", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--prepend-bos",
        type=parse_optional_bool,
        default=None,
        metavar="{auto,true,false}",
        help="Default: auto, using the SAE or model configuration.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    template_path = Path(args.prompt_template)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(dataset_path)
    required_columns = {
        "id",
        "environment_full",
        "unambiguous_direct",
    }
    missing = required_columns - set(rows.columns)
    if missing:
        raise KeyError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive.")

    rows = rows.sample(
        n=min(args.num_examples, len(rows)),
        random_state=args.dataset_seed,
    ).reset_index(drop=True)
    prompt_template = template_path.read_text(encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError(
            "This Gemma-9B evaluation requires a CUDA GPU."
        )

    print("Loading TransformerLens model...")
    model = HookedTransformer.from_pretrained(
        args.model,
        device=device,
        dtype=torch.float16,
    )

    print("Loading SAE...")
    loaded_sae = SAE.from_pretrained(
        release=args.sae_release,
        sae_id=args.sae_id,
        device=device,
    )
    sae = unpack_sae(loaded_sae)

    configured_hook = config_value(sae, "hook_name", None)
    if configured_hook is not None and configured_hook != args.hook_name:
        raise RuntimeError(
            "The loaded SAE hook does not match the reported run: "
            f"SAE says {configured_hook!r}, expected {args.hook_name!r}."
        )

    prepend_bos = (
        resolve_prepend_bos(sae, model)
        if args.prepend_bos is None
        else bool(args.prepend_bos)
    )

    if not hasattr(sae, "W_dec"):
        raise TypeError(
            f"Loaded SAE type {type(sae)!r} does not expose W_dec."
        )
    if not (0 <= args.feature < sae.W_dec.shape[0]):
        raise IndexError(
            f"Feature {args.feature} is outside SAE range "
            f"[0, {sae.W_dec.shape[0] - 1}]."
        )

    print("\nResolved settings")
    print(f"  model          = {args.model}")
    print(f"  hook           = {args.hook_name}")
    print(f"  prepend_bos    = {prepend_bos}")
    print(f"  feature        = {args.feature}")
    print(f"  strength       = {args.strength}")
    print(f"  max_act        = {args.max_act}")
    print(f"  examples       = {len(rows)}")
    print(f"  clear field    = unambiguous_direct\n")

    baseline_path = output_dir / "baseline_clear.json"
    steered_path = output_dir / "steered_clear.json"

    baseline_results = evaluate_condition(
        condition="baseline",
        model=model,
        sae=sae,
        rows=rows,
        prompt_template=prompt_template,
        hook_name=args.hook_name,
        feature=args.feature,
        strength=args.strength,
        max_act=args.max_act,
        prepend_bos=prepend_bos,
        max_new_tokens=args.max_new_tokens,
        generation_seed=args.generation_seed,
        checkpoint_path=baseline_path,
    )

    steered_results = evaluate_condition(
        condition="steered",
        model=model,
        sae=sae,
        rows=rows,
        prompt_template=prompt_template,
        hook_name=args.hook_name,
        feature=args.feature,
        strength=args.strength,
        max_act=args.max_act,
        prepend_bos=prepend_bos,
        max_new_tokens=args.max_new_tokens,
        generation_seed=args.generation_seed,
        checkpoint_path=steered_path,
    )

    summary = summarize(
        baseline_results,
        steered_results,
        bootstrap_seed=args.generation_seed + 100_000,
    )
    summary["configuration"] = {
        "model": args.model,
        "sae_release": args.sae_release,
        "sae_id": args.sae_id,
        "hook_name": args.hook_name,
        "feature": args.feature,
        "strength": args.strength,
        "max_act": args.max_act,
        "prepend_bos": prepend_bos,
        "dataset_seed": args.dataset_seed,
        "generation_seed": args.generation_seed,
        "input_field": "unambiguous_direct",
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Clear-input evaluation ===")
    print_binary_summary(
        "Over-asking: at least one question on a clear instruction",
        summary["overasking"],
    )
    print_binary_summary(
        "False-positive ambiguity classification",
        summary["predicted_ambiguous_false_positive"],
    )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
