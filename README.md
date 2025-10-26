# Evaluation pipeline for AmbiK dataset with SAE steering experiments

Files:
- `llm.py`: contains LLM classes
- `runner.py`: runs evaluation and writes results JSON
- `provider.py`: simple scripted provider
- `text_matching.py`: matching utilities
- `eval_metrics.py`: computes statistics from results 
- `data/`: contains data from [AmbiK](https://github.com/cog-model/AmbiK-dataset) 

## How to run

Install required dependencies:
```python
pip install sae-lens transformer-lens sae-dashboard sentence_transformers langchain_nebius
```

Usage example:
```python
python runner.py --model_name qwen --dataset_csv /path/to/AmbiK_data.csv --out_json results/output.json --num_examples 100 --mode both
```
Results are saved to `results/` folder.
Then compute metrics:

```python
from eval_metrics import compute_metrics_from_json, print_metrics
m = compute_metrics_from_json('results/output.json')
print_metrics(m)
```
