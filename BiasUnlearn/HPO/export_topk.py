# hpo/export_topk.py
from __future__ import annotations
import argparse
import csv
import optuna

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_biasunlearn.db")
    parser.add_argument("--study_name", type=str, default="biasunlearn_gpt2l_v1")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=str, default="reports/day2_topk.csv")
    args = parser.parse_args()

    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )

    trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    trials = sorted(trials, key=lambda x: x.value, reverse=True)[: args.topk]

    fieldnames = ["trial_number", "value", "params", "user_attrs"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            writer.writerow({
                "trial_number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            })

if __name__ == "__main__":
    main()