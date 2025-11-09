# Evaluation pipeline for AmbiK dataset with SAE steering experiments

Files:
- `llm.py`: contains LLM classes
- `runner.py`: runs evaluation and writes results to JSON
- `provider.py`: simple scripted provider
- `text_matching.py`: matching utilities
- `eval_metrics.py`: computes statistics from results 
- `data/`: contains data from [AmbiK](https://github.com/cog-model/AmbiK-dataset) 
- `utils/`: contains scripts for logging and model's output parsing

## How to run

Install required dependencies:
```python
pip install sae-lens transformer-lens sae-dashboard sentence_transformers langchain_nebius
```

Usage examples:
1. Loading any model from Huggingface (without steering)
    ```python
    !python runner.py \
    --model_name google/gemma-2b-it \
    --num_examples 10 \
    --mode proxy  \
    --dataset_csv data/ambik_calib_100.csv \
    --out_json results/ambik_eval_steered.json \
    --seed 123
    ```
2. Loading model via custom class (without steering). Custom class is used when flag `use_steering` is `True`.
    ```python
    !python runner.py \
    --model_name gemma-steered \
    --use_steering \
    --num_examples 10 \
    --mode proxy  \
    --dataset_csv data/ambik_calib_100.csv \
    --out_json results/ambik_eval.json \
    --seed 123
    ```
3. Loading model via custom class (with steering)
    ```python
    !python runner.py \
    --model_name gemma-steered \
    --use_steering \
    --num_examples 10 \
    --steering_feature 1909 \
    --steering_strength 2 \
    --max_act 4 \
    --compute_max_per_turn \
    --mode proxy  \
    --dataset_csv data/ambik_calib_100.csv \
    --out_json results/ambik_eval_steered.json \
    --seed 123
    ```
Results are saved to `results/` folder. 

> [!NOTE]  
> It is recommended to delete model cache across multiple runs so cache would not be reused.
>
> For example, `!rm -f log/{cache filename}.pkl`.

Then compute metrics:

```python
from eval_metrics import compute_metrics_from_json, print_metrics
m = compute_metrics_from_json('results/ambik_eval_steered.json.json')
print_metrics(m)
```

Example of output json:

```json
    {
    "total": 10,
    "counts_per_category": {
        "preferences": 6,
        "common_sense_knowledge": 4
    },
    "per_category_similarity": {
        "preferences": 0.32645830512046814,
        "common_sense_knowledge": 0.5074414809544882
    },
    "num_questions_hist": {
        "0": 5,
        "2": 3,
        "1": 2
    },
    "avg_num_questions": 0.8,
    "necessity_precision": 0.4,
    "necessity_recall": 0.3333333333333333,
    "resolved_proxy_rate": 0.1,
    "resolved_dialog_rate": null,
    "overall_weighted_score": 0.41068595091501875
    }
```

## What statistics are calculated?

The function `eval_metrics.py` aggregates a run into a metrics dictionary with:

- **total** — total number of examples.

- **counts_per_category** — count of examples per `ambiguity_type`.

- **per_category_similarity** — average of `model_question_best_similarity` **only on examples where the model asked ≥1 question**, reported per category (otherwise `null`).

- **num_questions_hist** — histogram of how many clarifying questions the model asked (keys: 0, 1, 2, …).

- **avg_num_questions** — mean number of questions per example (zeros included).

- **necessity_precision** — precision of “asking only when needed”  
  $$\text{TP} / (\text{TP} + \text{FP})$$,   
  where $TP$ = examples in the `"preferences"` category where the model asked ≥1 question;  
  $FP$ = examples **not** in `"preferences"` where the model asked ≥1 question.

- **necessity_recall** — recall of “asking when needed”  
  $$\text{TP} / (\text{TP} + \text{FN})$$,
  where $FN$ = `"preferences"` examples where the model **did not** ask a question.

- **resolved_proxy_rate** — fraction of examples marked `resolved_proxy = true`  
  (best similarity ≥ a threshold like 0.75).

- **resolved_dialog_rate** — among examples that have a dialog label (`dialog.resolved_dialog`), the fraction resolved successfully (`null` if none).

- **overall_weighted_score** — a single summary score:              
  $$0.5 \cdot \text{necessity} \textunderscore \text{score} ~ + ~ 0.4 \cdot \text{overall} \textunderscore \text{similarity} ~ + ~ 0.1 \cdot text{brevity} \textunderscore text{score} $$,      
  where  
  • **necessity_score** = $TP$ / (number of `"preferences"` examples) — effectively recall on that category;  
  • **overall_similarity** — average `model_question_best_similarity` across **all examples where ≥1 question was asked**;  
  • **brevity_score** — fraction of examples with questions ≤ `brevity_max` (default 1).
