# scripts/run_day2_workers.sh
#!/usr/bin/env bash
set -euo pipefail

GAP_BASE=${1:?need gap_base}
LM_BASE=${2:?need lm_base}

python -m hpo.worker --gpu_id 0 --n_trials 5 --gap_base ${GAP_BASE} --lm_base ${LM_BASE} &
python -m hpo.worker --gpu_id 1 --n_trials 5 --gap_base ${GAP_BASE} --lm_base ${LM_BASE} &
python -m hpo.worker --gpu_id 2 --n_trials 5 --gap_base ${GAP_BASE} --lm_base ${LM_BASE} &
python -m hpo.worker --gpu_id 3 --n_trials 5 --gap_base ${GAP_BASE} --lm_base ${LM_BASE} &
python -m hpo.worker --gpu_id 4 --n_trials 5 --gap_base ${GAP_BASE} --lm_base ${LM_BASE} &
python -m hpo.worker --gpu_id 5 --n_trials 5 --gap_base ${GAP_BASE} --lm_base ${LM_BASE} &

wait