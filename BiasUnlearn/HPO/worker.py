'''
读取共享 study
在固定 GPU 上取一个 trial
启动 CUDA_VISIBLE_DEVICES=<gpu_id> accelerate launch train.py ...
每隔 10–15 秒读一次该 trial 的 metrics.jsonl
trial.report(score, eval_idx)
触发 trial.should_prune() 时杀掉子进程并标记 prune
'''

# hpo/worker.py
from __future__ import annotations
import argparse
import os
import subprocess
import time
from pathlib import Path

import optuna

from hpo.objective import load_last_eval, compute_objective, compute_gap_stats, should_hard_prune

def get_study(storage: str, study_name: str) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42,
            multivariate=True,
            n_startup_trials=10,
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=4,
            reduction_factor=2,
        ),
    )

def sample_config(trial: optuna.Trial) -> dict:
    lr = trial.suggest_float("lr", 1e-5, 8e-5, log=True)
    beta = trial.suggest_float("beta", 0.05, 0.4)

    a1 = trial.suggest_float("a1_raw", 0.1, 2.0, log=True)
    a2 = trial.suggest_float("a2_raw", 0.1, 2.0, log=True)
    a3 = trial.suggest_float("a3_raw", 0.05, 1.0, log=True)
    s = a1 + a2 + a3
    ster_weight = a1 / s
    anti_weight = a2 / s
    kl_weight = a3 / s

    return {
        "lr": lr,
        "beta": beta,
        "ster_weight": ster_weight,
        "anti_weight": anti_weight,
        "kl_weight": kl_weight,
        "max_unlearn_steps": trial.suggest_categorical("max_unlearn_steps", [300, 500, 800, 1000]),
        "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16]),
        "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32]),
        "lora_dropout": trial.suggest_categorical("lora_dropout", [0.0, 0.05, 0.1]),
    }

def build_cmd(cfg: dict, trial_number: int, gpu_id: int, model_name: str) -> tuple[list[str], str, str]:
    exp_name = f"optuna_t{trial_number:04d}"
    save_dir = exp_name
    metrics_path = f"logs/{exp_name}_metrics.jsonl"

    cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} accelerate launch train.py \
      --model_name {model_name} \
      --use_lora \
      --fan_in_fan_out \
      --model_save_dir {save_dir} \
      --log_file logs/{exp_name}.log \
      --lr {cfg["lr"]} \
      --beta {cfg["beta"]} \
      --max_unlearn_steps {cfg["max_unlearn_steps"]} \
      --save_every 50 \
      --ster_batch_size 4 \
      --batch_size 28 \
      --ster_weight {cfg["ster_weight"]} \
      --anti_weight {cfg["anti_weight"]} \
      --kl_weight {cfg["kl_weight"]} \
      --mix_anti \
      --lora_r {cfg["lora_r"]} \
      --lora_alpha {cfg["lora_alpha"]} \
      --lora_dropout {cfg["lora_dropout"]}
    """.strip()

    return ["bash", "-lc", cmd], exp_name, metrics_path

def objective_factory(gpu_id: int, model_name: str, gap_base: float, lm_base: float):
    def objective(trial: optuna.Trial) -> float:
        cfg = sample_config(trial)
        cmd, exp_name, metrics_path = build_cmd(cfg, trial.number, gpu_id, model_name)

        Path("log").mkdir(exist_ok=True, parents=True)
        Path("save").mkdir(exist_ok=True, parents=True)

        proc = subprocess.Popen(cmd)
        seen_steps = set()
        last_score = None

        try:
            while proc.poll() is None:
                m = load_last_eval(metrics_path)
                if m is not None:
                    step = int(m["step"])
                    if step not in seen_steps:
                        seen_steps.add(step)

                        hard_prune, reason = should_hard_prune(m, lm_base=lm_base)
                        if hard_prune:
                            trial.set_user_attr("hard_prune_reason", reason)
                            proc.kill()
                            raise optuna.TrialPruned()

                        score = compute_objective(m, gap_base=gap_base, lm_base=lm_base)
                        last_score = score

                        trial.report(score, step=len(seen_steps))
                        if trial.should_prune():
                            trial.set_user_attr("hard_prune_reason", "optuna_pruner")
                            proc.kill()
                            raise optuna.TrialPruned()

                time.sleep(15)

            if last_score is None:
                raise RuntimeError(f"No eval metrics found for trial {trial.number}")

            trial.set_user_attr("exp_name", exp_name)
            return last_score

        finally:
            if proc.poll() is None:
                proc.kill()

    return objective

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--n_trials", type=int, default=6)
    parser.add_argument("--study_name", type=str, default="biasunlearn_gpt2l_v1")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_biasunlearn.db")
    parser.add_argument("--model_name", type=str, default="/modelpath/gpt2-large")
    parser.add_argument("--gap_base", type=float, required=True)
    parser.add_argument("--lm_base", type=float, required=True)
    args = parser.parse_args()

    study = get_study(args.storage, args.study_name)
    objective = objective_factory(
        gpu_id=args.gpu_id,
        model_name=args.model_name,
        gap_base=args.gap_base,
        lm_base=args.lm_base,
    )
    study.optimize(objective, n_trials=args.n_trials)

if __name__ == "__main__":
    main()