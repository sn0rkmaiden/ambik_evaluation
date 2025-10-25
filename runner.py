import argparse, json, os, re
import pandas as pd
from llm import CustomLLM, HuggingFaceLLM
from provider import ProviderAgent
from text_matching import best_match_score, normalize_text
import ast

def extractJSON(raw_output):
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    return match.group(0)

def data2prompt(environment, ambiguous):
    prompt = "\n".join(open("data/prompt_draft.txt").readlines())
    prompt = prompt.replace('<DESCRIPTION>', environment)
    prompt = prompt.replace('<TASK>', ambiguous)
    return prompt

def get_model_questions(model, instruction, max_q=3):

    instruction += "\nReturn **only** a valid JSON object without any extra text."

    out, _ = model.request(instruction, None, json_format=True)

    print(f"model's output\n{out}")
    
    out = extractJSON(out)
    print(f"postprocess: {out}")

    try:
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

def run_eval(dataset_csv='data/ambik_calib_100.csv', out_json='results/ambik_eval_output.json', num_examples=None, seed=0, mode='proxy', model=None):

    print(">>> Reading data")
    df = pd.read_csv(dataset_csv)
    if num_examples is not None:
        df = df.sample(n=min(num_examples, len(df)), random_state=seed).reset_index(drop=True)

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

        # get model questions
        if model is None:
            model_questions = []
        else:
            res = get_model_questions(model, instruction)
            model_questions = res['question']
            if not isinstance(model_questions, list):
                model_questions = []

            is_amb = res['ambiguous'] # TODO add metric for checking binary ambiguity

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

        # dialog mode: provider replies with gold answer, then model optionally produces final action
        example['dialog'] = None
        if mode in ('dialog', 'both'):
            prov_reply = provider.reply(example['gold_answer'])
            dialog_record = {'provider_answers': [prov_reply], 'model_final_action': None, 'resolved_dialog': None}
            # try to get final action from model via generate_final_action or similar
            final_action = None
            if model is not None:
                if hasattr(model, 'generate_final_action'):
                    final_action = model.generate_final_action(ambiguous, history=[prov_reply])
                elif hasattr(model, 'generate_action'):
                    final_action = model.generate_action(ambiguous, history=[prov_reply])
                else:
                    # no method: leave None
                    final_action = None
            dialog_record['model_final_action'] = final_action
            # check resolution against gold plan_for_clear
            if final_action:
                dialog_record['resolved_dialog'] = best_match_score(final_action, example['gold_plan_for_clear']) >= 0.75
            example['dialog'] = dialog_record

        results.append(example)
    # save JSON
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_json} ({len(results)} examples)")
    return out_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--num_examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=['proxy','dialog','both'], default='both')
    parser.add_argument('--dataset_csv', required=False)
    parser.add_argument('--out_json', required=False)
    args = parser.parse_args()

    print(">>> Loading LLM")
    model_name = args.model_name
    if model_name == "qwen":
        model = CustomLLM(model_name, cache=f'log/{model_name}_cache.pkl')
    elif 'gemma' in model_name.lower():
        model = HuggingFaceLLM(model_name, cache="log/gemma_cache.pkl")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if args.dataset_csv is None:
        dataset_path = "data/ambik_calib_100.csv"
    else:
        dataset_path = args.dataset_csv
    
    if args.out_json is None:
        output_file = "results/ambik_eval_output.json"
    else:
        output_file = args.out_json

    run_eval(dataset_path, output_file, num_examples=args.num_examples, seed=args.seed, mode=args.mode, model=model)
