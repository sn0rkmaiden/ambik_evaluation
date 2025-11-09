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

