#!/usr/bin/env python3
"""
Evaluate over-asking on AmbiK's paired unambiguous instructions.

Run this file from the root of the ambik_evaluation repository, where llm.py,
utils/parsing.py, data/prompt_draft.txt, and data/ambik_calib_100.csv exist.

The script loads Gemma-2-9B and its SAE once, evaluates the same 100 clear
instructions under:
  1) no steering
  2) feature 166, strength 30, max_act 4

It reports the false-positive/over-asking rate, Wilson 95% confidence intervals,
a paired difference confidence interval, and an exact McNemar p-value.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from llm import HookedGEMMA
from utils.parsing import parse_model_json


Z_975 = 1.959963984540054


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_prompt(environment: str, task: str, template_path: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    return template.replace("<DESCRIPTION>", str(environment)).replace("<TASK>", str(task))


def generate_decision(model: HookedGEMMA, prompt: str) -> tuple[bool, list[str], str]:
    # Match runner.py's instruction and JSON parsing behavior.
    instruction = prompt + "\nReturn **only** a valid JSON object without any extra text."
    raw, _ = model.request(instruction, None, json_format=True)
    parsed = parse_model_json(raw)
    if parsed is None:
        raise ValueError(f"Could not parse model output:\n{raw}")

    predicted_ambiguous = bool(parsed.get("ambiguous", False))
    questions = parsed.get("question", parsed.get("questions", []))
    if not isinstance(questions, list):
        questions = []
    questions = [str(q).strip() for q in questions if str(q).strip()]
    return predicted_ambiguous, questions, raw


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    z = Z_975
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(baseline_only: int, steered_only: int) -> float:
    n = baseline_only + steered_only
    if n == 0:
        return 1.0
    k = min(baseline_only, steered_only)
    lower_tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * lower_tail)


def paired_bootstrap_ci(
    baseline: np.ndarray,
    steered: np.ndarray,
    seed: int,
    n_bootstrap: int = 20000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    diffs = steered - baseline
    estimates = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, 2000):
        stop = min(start + 2000, n_bootstrap)
        idx = rng.integers(0, n, size=(stop - start, n))
        estimates[start:stop] = diffs[idx].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate_condition(
    model: HookedGEMMA,
    rows: pd.DataFrame,
    *,
    label: str,
    steering_feature: int | None,
    steering_strength: float,
    max_act: float | None,
    seed: int,
    prompt_template: Path,
) -> list[dict[str, Any]]:
    model.steering_feature = steering_feature
    model.steering_strength = float(steering_strength)
    model.max_act = max_act
    model.compute_max_per_turn = False
    model.steering_features = []

    output: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(rows.iterrows()):
        # Use the same per-example random seed in baseline and steered conditions.
        set_seed(seed + position)
        prompt = make_prompt(
            row.get("environment_full", ""),
            row.get("unambiguous_direct", ""),
            prompt_template,
        )
        predicted_ambiguous, questions, raw = generate_decision(model, prompt)
        output.append(
            {
                "id": int(row.get("id", position)),
                "condition": label,
                "instruction": str(row.get("unambiguous_direct", "")),
                "predicted_ambiguous": predicted_ambiguous,
                "model_questions": questions,
                "asked_question": len(questions) > 0,
                "num_questions": len(questions),
                "raw_output": raw,
            }
        )
        print(
            f"[{label}] {position + 1:03d}/{len(rows)} "
            f"id={output[-1]['id']} asked={output[-1]['asked_question']}"
        )
    return output


def summarize_binary_pair(
    baseline_values: np.ndarray,
    steered_values: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, Any]:
    n = len(baseline_values)
    b_count = int(baseline_values.sum())
    s_count = int(steered_values.sum())
    b_ci = wilson_interval(b_count, n)
    s_ci = wilson_interval(s_count, n)
    diff_ci = paired_bootstrap_ci(
        baseline_values,
        steered_values,
        bootstrap_seed,
    )
    baseline_only = int(
        np.sum((baseline_values == 1) & (steered_values == 0))
    )
    steered_only = int(
        np.sum((baseline_values == 0) & (steered_values == 1))
    )
    return {
        "baseline_count": b_count,
        "baseline_rate": b_count / n,
        "baseline_wilson_95ci": list(b_ci),
        "steered_count": s_count,
        "steered_rate": s_count / n,
        "steered_wilson_95ci": list(s_ci),
        "paired_change": float(
            steered_values.mean() - baseline_values.mean()
        ),
        "paired_change_bootstrap_95ci": list(diff_ci),
        "baseline_only": baseline_only,
        "steered_only": steered_only,
        "exact_mcnemar_p": exact_mcnemar_p(
            baseline_only,
            steered_only,
        ),
    }


def summarize(
    baseline: list[dict[str, Any]],
    steered: list[dict[str, Any]],
    bootstrap_seed: int,
) -> dict[str, Any]:
    b_by_id = {int(x["id"]): x for x in baseline}
    s_by_id = {int(x["id"]): x for x in steered}
    ids = sorted(set(b_by_id) & set(s_by_id))

    b_ask = np.array(
        [bool(b_by_id[i]["asked_question"]) for i in ids],
        dtype=float,
    )
    s_ask = np.array(
        [bool(s_by_id[i]["asked_question"]) for i in ids],
        dtype=float,
    )
    b_amb = np.array(
        [bool(b_by_id[i]["predicted_ambiguous"]) for i in ids],
        dtype=float,
    )
    s_amb = np.array(
        [bool(s_by_id[i]["predicted_ambiguous"]) for i in ids],
        dtype=float,
    )

    return {
        "n": len(ids),
        "overasking": summarize_binary_pair(
            b_ask,
            s_ask,
            bootstrap_seed,
        ),
        "predicted_ambiguous_false_positive": summarize_binary_pair(
            b_amb,
            s_amb,
            bootstrap_seed + 1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/ambik_calib_100.csv")
    parser.add_argument("--prompt-template", default="data/prompt_draft.txt")
    parser.add_argument("--output-dir", default="results/clear_input_eval")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--model", default="gemma-2-9b-it")
    parser.add_argument("--sae-release", default="gemma-scope-9b-it-res-canonical")
    parser.add_argument("--sae-id", default="layer_20/width_16k/canonical")
    parser.add_argument("--feature", type=int, default=166)
    parser.add_argument("--strength", type=float, default=30.0)
    parser.add_argument("--max-act", type=float, default=4.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    prompt_path = Path(args.prompt_template)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(dataset_path)
    required = {"id", "environment_full", "unambiguous_direct"}
    missing = required - set(rows.columns)
    if missing:
        raise KeyError(f"Dataset is missing required columns: {sorted(missing)}")

    if args.num_examples is not None:
        rows = rows.sample(
            n=min(args.num_examples, len(rows)),
            random_state=args.seed,
        ).reset_index(drop=True)

    print("Loading model and SAE once...")
    model = HookedGEMMA(
        model_name=args.model,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        cache=None,  # Important: prevents baseline outputs being reused for steering.
        max_new_tokens=100,
        steering_feature=None,
        steering_strength=1.0,
        max_act=None,
        compute_max_per_turn=False,
    )

    # Compatibility with newer sae_lens releases, where SAE.from_pretrained()
    # returns a tuple such as (sae, config, sparsity) instead of only the SAE.
    if isinstance(model.sae, tuple):
        if not model.sae:
            raise RuntimeError("SAE.from_pretrained() returned an empty tuple.")
        model.sae = model.sae[0]

    if not hasattr(model.sae, "cfg"):
        raise TypeError(
            "The loaded SAE object does not expose `.cfg`. "
            f"Received type: {type(model.sae)!r}"
        )

    baseline = evaluate_condition(
        model,
        rows,
        label="baseline",
        steering_feature=None,
        steering_strength=1.0,
        max_act=None,
        seed=args.seed,
        prompt_template=prompt_path,
    )
    (out_dir / "baseline_clear.json").write_text(
        json.dumps({"examples": baseline}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    steered = evaluate_condition(
        model,
        rows,
        label="steered",
        steering_feature=args.feature,
        steering_strength=args.strength,
        max_act=args.max_act,
        seed=args.seed,
        prompt_template=prompt_path,
    )
    (out_dir / "steered_clear.json").write_text(
        json.dumps({"examples": steered}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = summarize(baseline, steered, bootstrap_seed=args.seed + 100000)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== Clear-input false positives ===")
    for title, key in [
        ("Over-asking (at least one question)", "overasking"),
        ("Predicted ambiguous", "predicted_ambiguous_false_positive"),
    ]:
        result = summary[key]
        print(f"\n{title}")
        print(
            f"Baseline: {result['baseline_count']}/{summary['n']} "
            f"= {result['baseline_rate']:.3f} "
            f"[{result['baseline_wilson_95ci'][0]:.3f}, "
            f"{result['baseline_wilson_95ci'][1]:.3f}]"
        )
        print(
            f"Steered:  {result['steered_count']}/{summary['n']} "
            f"= {result['steered_rate']:.3f} "
            f"[{result['steered_wilson_95ci'][0]:.3f}, "
            f"{result['steered_wilson_95ci'][1]:.3f}]"
        )
        print(
            f"Change:   {result['paired_change']:+.3f} "
            f"[{result['paired_change_bootstrap_95ci'][0]:+.3f}, "
            f"{result['paired_change_bootstrap_95ci'][1]:+.3f}]"
        )
        print(f"Exact McNemar p = {result['exact_mcnemar_p']:.6g}")

    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
