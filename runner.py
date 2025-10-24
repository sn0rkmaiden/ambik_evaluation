
import argparse, json, os, random, time
import pandas as pd
from llm import LLM, CustomLLM, HuggingfaceLLM
from provider import ProviderAgent
from text_matching import best_match_score, normalize_text

def get_model_questions(model, instruction, context=None, max_q=3):
    # Try to call several possible interfaces; keep it simple.
    if hasattr(model, "generate_questions"):
        out = model.generate_questions(instruction, context)
    elif hasattr(model, "ask"):
        out = model.ask(instruction)
    elif hasattr(model, "do"):
        try:
            out, _ = model.do(instruction)
        except Exception:
            out = model.do(instruction)
    elif hasattr(model, "request"):
        out = model.request(instruction)
    else:
        # fallback: if model has __call__, use it
        if callable(model):
            out = model(instruction)
        else:
            out = []
    # normalize to list
    if out is None:
        return []
    if isinstance(out, str):
        return [out]
    if isinstance(out, (list, tuple)):
        return list(out)[:max_q]
    return [str(out)]

def run_eval(dataset_csv, out_json, num_examples=None, seed=0, mode='proxy', model=None):
    df = pd.read_csv(dataset_csv)
    if num_examples is not None:
        df = df.sample(n=min(num_examples, len(df)), random_state=seed).reset_index(drop=True)
    results = []
    provider = ProviderAgent()
    for _, row in df.iterrows():
        example = {}
        example['id'] = int(row.get('id', -1)) if 'id' in row else -1
        example['ambiguity_type'] = row.get('ambiguity_type', 'unknown')
        ambiguous = row.get('ambiguous_task') or row.get('ambiguous_instruction') or row.get('instruction') or ''
        example['ambiguous_instruction'] = ambiguous
        example['gold_question'] = row.get('question', '') if 'question' in row else ''
        example['gold_answer'] = row.get('answer', '') if 'answer' in row else ''
        example['gold_plan_for_clear'] = row.get('plan_for_clear_task', '') if 'plan_for_clear_task' in row else row.get('plan_for_clear', '') or row.get('plan_for_clear_task', '') or ''
        # get model questions
        if model is None:
            model_questions = []
        else:
            model_questions = get_model_questions(model, ambiguous)
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
    parser.add_argument('--dataset_csv', required=True)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--num_examples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=['proxy','dialog','both'], default='both')
    args = parser.parse_args()
    # Use a very simple fallback model if CustomLLM not available/instantiable
    try:
        dummy = CustomLLM(cache=None)
    except Exception:
        class DummyModel:
            def generate_questions(self, instruction, context=None):
                return []
            def generate_final_action(self, instruction, history=None):
                return None
        dummy = DummyModel()
    run_eval(args.dataset_csv, args.out_json, num_examples=args.num_examples, seed=args.seed, mode=args.mode, model=dummy)
