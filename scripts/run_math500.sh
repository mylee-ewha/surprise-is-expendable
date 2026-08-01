# scripts/run_math500.sh
#!/bin/bash
cd "$(dirname "$0")/.."
conda activate cot-quant
python experiments/math500_ablation.py