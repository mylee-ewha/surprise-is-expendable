# scripts/run_gpqa.sh
#!/bin/bash
cd "$(dirname "$0")/.."
conda activate cot-quant
python experiments/gpqa_ablation.py