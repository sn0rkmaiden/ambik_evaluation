
ambik_eval - evaluation scripts for AmbiK dataset
Files:
- llm.py: (imported from ClarQ-LLM llm.py as requested) contains LLM classes like CustomLLM and HuggingfaceLLM.
- runner.py: runs evaluation and writes results JSON
- provider.py: simple scripted provider
- text_matching.py: matching utilities (exact/contains then embedding fallback)
- eval_metrics.py: computes statistics from results JSON

Usage example:
python runner.py --dataset_csv /path/to/AmbiK_data.csv --out_json results.json --num_examples 100 --mode both
Then compute metrics:
from eval_metrics import compute_metrics_from_json, print_metrics
m = compute_metrics_from_json('results.json')
print_metrics(m)
