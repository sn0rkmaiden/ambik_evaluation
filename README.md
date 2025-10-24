# Evaluation pipeline for AmbiK dataset with SAE steering experiments

Files:
- llm.py: contains LLM classes
- runner.py: runs evaluation and writes results JSON
- provider.py: simple scripted provider
- text_matching.py: matching utilities
- eval_metrics.py: computes statistics from results JSON

Usage example:
```python

    python runner.py --dataset_csv /path/to/AmbiK_data.csv --out_json results.json --num_examples 100 --mode both

Then compute metrics:

    from eval_metrics import compute_metrics_from_json, print_metrics
    m = compute_metrics_from_json('results.json')
    print_metrics(m)

```
