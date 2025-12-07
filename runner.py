import argparse, json, os, re, torch, gc
import pandas as pd
from llm import CustomLLM, HuggingFaceLLM, HookedGEMMA
from provider import ProviderAgent
from text_matching import best_match_score, normalize_text
import ast
from utils.parsing import *

from dotenv import load_dotenv
load_dotenv()

def summarize_model(m):
    info = {"wrapper_class": type(m).__name__}
    # common attributes across wrappers
    if hasattr(m, "model_name"):
        info["model_name"] = getattr(m, "model_name")
    if hasattr(m, "sae"):
        sae = getattr(m, "sae")
        info["sae_present"] = sae is not None
    if hasattr(m, "max_new_tokens"):
        info["max_new_tokens"] = getattr(m, "max_new_tokens")
    if hasattr(m, "cache_path"):
        info["cache_path"] = getattr(m, "cache_path")
    if hasattr(m, "device"):
        info["device"] = str(getattr(m, "device"))
    if hasattr(m, "steering_feature"):
        info["steering_feature"] = getattr(m, "steering_feature")
    if hasattr(m, "steering_strength"):
        info["steering_strength"] = getattr(m, "steering_strength")
    return info


def steering_summary(m):
    # works whether attrs exist or not
    used = bool(getattr(m, "steering_feature", None) is not None)
    cfg = None
    if used:
        sae = getattr(m, "sae", None)
        cfg = {
            "feature": int(getattr(m, "steering_feature")),
            "strength": float(getattr(m, "steering_strength", 1.0)),
            "max_act": (
                float(getattr(m, "max_act"))
                if getattr(m, "max_act", None) is not None
                else None
            ),
            "compute_max_per_turn": bool(
                getattr(m, "compute_max_per_turn", False)
            ),
            "sae_release": getattr(getattr(sae, "cfg", None), "release", None)
            if sae
            else None,
            "sae_id": (
                getattr(
                    getattr(getattr(sae, "cfg", None), "metadata", None),
                    "hook_name",
                    None,
                )
                if sae
                else None
            ),
        }
    return used, cfg


def parse_steering_features(args) -> list[int]:
    """
    Parse steering features from CLI args.

    Priority:
    1) --steering_features (file or comma/space-separated list)
    2) --steering_feature (single int)

    Returns a *unique, sorted* list of ints (may be empty).
    """
    feats: list[int] = []

    raw = getattr(args, "steering_features", None)
    if raw:
        raw = raw.strip()
        if os.path.exists(raw):
            if raw.endswith(".json"):
                with open(raw, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    feats = [int(k) for k in data.keys()]
                elif isinstance(data, list):
                    feats = [int(x) for x in data]
                else:
                    raise ValueError(
                        f"Unsupported JSON structure in {raw} for steering features"
                    )
            else:                
                with open(raw, "r") as f:
                    txt = f.read()
                feats = [int(m.group()) for m in re.finditer(r"-?\d+", txt)]
        else:
            tokens = re.split(r"[,\s]+", raw)
            feats = [int(t) for t in tokens if t]

    elif args.steering_feature is not None:
        feats = [int(args.steering_feature)]

    feats = sorted(set(feats))
    return feats

def build_final_prompt(instruction: str, clarifying_questions: list[str], provider_reply: str | None) -> str:
    """
    Build the final prompt for the robot model to generate a plan of actions.
    """

    prompt_parts = [
        "You are a service robot operating in an office kitchen.",
        f"Your initial instruction was: '{instruction.strip()}'."
    ]

    if clarifying_questions:
        questions_text = "\n".join(
            [f"{i+1}. {q.strip()}" for i, q in enumerate(clarifying_questions) if q.strip()]
        )
        prompt_parts.append(
            "To better understand the task, you asked the following clarifying question(s):\n" + questions_text
        )
        if provider_reply:
            prompt_parts.append(f"The human responded with: '{provider_reply.strip()}'")
        prompt_parts.append(
            "Now that the ambiguity has been resolved, generate your final plan of actions as the robot. "
            "Be specific about what you will do to execute the clarified instruction. "
            "Return only the plan, without any additional explanations or comments."
        )
    else:
        prompt_parts.append(
            "The instruction is clear and requires no clarification. "
            "Generate your plan of actions as the robot. "
            "Be specific about what you will do to execute the instruction. "
            "Return only the plan, without any additional explanations or comments."
        )

    return "\n\n".join(prompt_parts)


def extractJSON(raw_output):
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    return match.group(0) if match is not None else "" 

def data2prompt(environment, ambiguous):
    prompt = "\n".join(open("data/prompt_draft.txt").readlines())
    prompt = prompt.replace('<DESCRIPTION>', environment)
    prompt = prompt.replace('<TASK>', ambiguous)
    return prompt

def get_model_questions2(model, instruction, max_q=3):

    instruction += "\nReturn **only** a valid JSON object without any extra text."

    out, _ = model.request(instruction, None, json_format=True)

    # print(f"model's output\n{out}")
    
    out = extractJSON(out)
    print(f"processed output: {out}")

    try:
        if out == "": return ""
        return json.loads(out)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(out)
            return obj
        except Exception as e:
            print("JSON decode error:", e)
            print("Prompt was:\n", instruction)
            print("Raw response:\n", repr(out))
            raise

def get_model_questions(model, instruction, max_q=3):
    instruction += "\nReturn **only** a valid JSON object without any extra text."

    out, _ = model.request(instruction, None, json_format=True)

    # print('------------RAW OUTPUT--------------')
    # print(out)
    # print('------------RAW OUTPUT--------------')

    obj = parse_model_json(out)
    if obj is None:
        print("JSON decode error.")
        print("Prompt was:\n", instruction)
        print("Raw response:\n", repr(out))
        # You can choose to raise, or return a default
        raise ValueError("Failed to parse model JSON.")
    
    obj.setdefault("ambiguous", False)
    obj.setdefault("question", [])
    return obj


def run_eval(dataset_csv='data/ambik_calib_100.csv', 
             out_json='results/ambik_eval_output.json', 
             num_examples=None, 
             seed=0, 
             mode='proxy', 
             model=None):

    print(">>> Reading data")
    df = pd.read_csv(dataset_csv)
    if num_examples is not None:
        df = df.sample(n=min(num_examples, len(df)), random_state=seed).reset_index(drop=True)

    model_info = summarize_model(model) if model is not None else {"wrapper_class": None}
    steering_used, steering_cfg = steering_summary(model) if model is not None else (False, None)

    results = []
    provider = ProviderAgent()

    for _, row in df.iterrows():
        example = {}
        example['id'] = int(row.get('id', -1)) if 'id' in row else -1
        example['ambiguity_type'] = row.get('ambiguity_type', 'unknown')
        env = row.get('environment_full')
        example['environment'] = env
        ambiguous = row.get('ambiguous_task')
        example['ambiguous_instruction'] = ambiguous
        example['gold_question'] = row.get('question', '') if 'question' in row else ''
        example['gold_answer'] = row.get('answer', '') if 'answer' in row else ''
        example['gold_plan_for_clear'] = row.get('plan_for_clear_task', '') if 'plan_for_clear_task' in row else ''     

        instruction = data2prompt(env, ambiguous)

        print(f"TYPE: {example['ambiguity_type']}")

        # get model questions
        if model is None:
            model_questions = []
        else:
            res = get_model_questions(model, instruction)
            if res != "":
                model_questions = res.get('question', res.get('questions', ""))
            else: 
                model_questions = ""
            if not isinstance(model_questions, list):
                model_questions = []

            is_amb = res.get('ambiguous', "") if res != "" else "" # TODO add metric for checking binary ambiguity

        example['model_questions'] = model_questions
        example['num_questions'] = len(model_questions)

        # compute best similarity between any model question and gold question
        best_score = 0.0
        for q in model_questions:
            s = best_match_score(q, example['gold_question'])
            if s > best_score:
                best_score = s

        example['model_question_best_similarity'] = best_score
        example['resolved_proxy'] = best_score >= 0.75

        example['dialog'] = None
        if mode in ('dialog', 'both') and model_questions:
            prov_reply = provider.reply(example['gold_answer'])
            dialog_record = {'provider_answers': [prov_reply], 'model_final_action': None, 'resolved_dialog': None}
            final_action = None
            if model is not None:
                prompt = build_final_prompt(instruction, model_questions, prov_reply)
                final_action = model.request(prompt, None, json_format=True)
            dialog_record['model_final_action'] = final_action
            if final_action:
                dialog_record['resolved_dialog'] = best_match_score(final_action, example['gold_plan_for_clear']) >= 0.75
            example['dialog'] = dialog_record

        results.append(example)
    
    payload = {
            "run_info": {
                "dataset_csv": dataset_csv,
                "output_json": out_json,
                "seed": seed,
                "mode": mode,
                "num_examples": num_examples,
                "model": model_info,
                "steering_used": steering_used,
                "steering_cfg": steering_cfg,
            },
            "examples": results
        }
    
    # Create parent directory of out_json if needed
    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"Saved results to {out_json} ({len(results)} examples)")
    return out_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--num_examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=['proxy','dialog','both'], default='both')
    parser.add_argument('--dataset_csv', required=False)
    parser.add_argument('--out_json', required=False)
    parser.add_argument('--use_steering', action='store_true',
                        help='Enable SAE steering for Gemma (HookedTransformer path)')
    parser.add_argument('--steering_feature', type=int, default=None,
                        help='SAE feature index to steer (int)')
    parser.add_argument('--steering_features', type=str, default=None,
                    help='Either comma/space-separated feature ids '
                            'or a path to a file (txt/json) containing feature ids.')
    parser.add_argument('--steering_strength', type=float, default=1.0,
                        help='Steering strength multiplier')
    parser.add_argument('--sae_release', type=str, default='gemma-2b-it-res-jb',
                        help='SAE release name for SAE.from_pretrained')
    parser.add_argument('--sae_id', type=str, default='blocks.12.hook_resid_post',
                        help='SAE id / hook name for SAE.from_pretrained')
    parser.add_argument('--max_act', type=float, default=None,
                        help='Fixed max activation to use (skip per-prompt estimation)')
    parser.add_argument('--compute_max_per_turn', action='store_true',
                        help='Estimate max activation from each prompt (no ActivationStore needed)')
    parser.add_argument('--gemma_model', type=str, default='gemma-2b-it',
                        help='HookedTransformer checkpoint name (e.g., gemma-2b-it)')

    args = parser.parse_args()

    model_name = args.model_name.strip()
    feature_list = parse_steering_features(args)

    # --- IO paths ---
    if args.dataset_csv is None:
        dataset_path = "data/ambik_calib_100.csv"
    else:
        dataset_path = args.dataset_csv

    if args.out_json is None:
        base_output_file = "results/ambik_eval_output.json"
    else:
        base_output_file = args.out_json

     # --- Multi-feature steering mode ---
    if len(feature_list) > 1:
        if 'gemma' not in model_name.lower():
            raise ValueError(
                "Multiple steering features are only supported for Gemma models. "
                "Got model_name={model_name!r}."
            )

        print(f">>> Multi-feature steering mode with features: {feature_list}")

        root, ext = os.path.splitext(base_output_file)
        if not ext:
            ext = ".json"

        for fid in feature_list:
            out_json = f"{root}_feat{fid}_str{args.steering_strength}{ext}"
            cache_path = f"log/gemma_cache_feat{fid}.pkl"

            print(f"\n>>> Loading HookedGEMMA for feature {fid}")
            model = HookedGEMMA(
                model_name=args.gemma_model,
                sae_release=args.sae_release,
                sae_id=args.sae_id,
                cache=cache_path,
                max_new_tokens=100,
                steering_feature=fid,
                steering_strength=args.steering_strength,
                max_act=(args.max_act if args.max_act is not None else None),
                compute_max_per_turn=args.compute_max_per_turn,
            )

            print(f">>> Running evaluation for feature {fid}, saving to {out_json}")
            run_eval(
                dataset_path,
                out_json,
                num_examples=args.num_examples,
                seed=args.seed,
                mode=args.mode,
                model=model,
            )

            del model
            gc.collect()
            torch.cuda.empty_cache()

            if os.path.exists(cache_path):
                os.remove(cache_path)

        return
    
    # Single-feature or no-steering mode
    
    print(">>> Loading LLM")

    use_steering_flag = args.use_steering or (len(feature_list) == 1)

    if model_name.lower() == "qwen":
        model = CustomLLM(model_name, cache=f'log/{model_name}_cache.pkl')

    elif 'gemma' in model_name.lower() and use_steering_flag:
        steering_feature = feature_list[0] if feature_list else args.steering_feature

        root, ext = os.path.splitext(base_output_file)
        output_file = f"{root}_feat{steering_feature}{ext}"

        print(f">>> Running evaluation for feature {steering_feature}, saving to {output_file}")

        model = HookedGEMMA(
            model_name=args.gemma_model,
            sae_release=args.sae_release,
            sae_id=args.sae_id,
            cache="log/gemma_cache.pkl",
            max_new_tokens=100,
            steering_feature=steering_feature,
            steering_strength=args.steering_strength,
            max_act=(args.max_act if args.max_act is not None else None),
            compute_max_per_turn=args.compute_max_per_turn,
        )

    elif 'gemma' in model_name.lower():
        model = HuggingFaceLLM(model_name, cache="log/gemma_cache.pkl")

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Single run output
    if not output_file:
        output_file = base_output_file

    run_eval(
        dataset_path,
        output_file,
        num_examples=args.num_examples,
        seed=args.seed,
        mode=args.mode,
        model=model,
    )

    return


if __name__ == "__main__":
    main()
