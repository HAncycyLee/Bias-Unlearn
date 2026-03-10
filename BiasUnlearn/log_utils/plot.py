import re
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_report(text: str):
    config = {}
    train_rows = []
    eval_rows = []

    # -------------------------
    # 1) 解析 Config
    # -------------------------
    config_pattern = re.compile(r"^\[Config\]\s+([^:]+):\s*(.*)$")
    for line in text.splitlines():
        m = config_pattern.match(line.strip())
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            config[key] = value

    # -------------------------
    # 2) 解析训练 batch 行
    # -------------------------
    train_pattern = re.compile(
        r"batch:\s*(?P<batch>\d+),\s*"
        r"lr:\s*(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]+"
        r"ster_lm:\s*(?P<ster_lm>[-+]?\d*\.?\d+)[,\s]+"
        r"ster_ref_lm:\s*(?P<ster_ref_lm>[-+]?\d*\.?\d+)[,\s]+"
        r"neg_log_ratio:\s*(?P<neg_log_ratio>[-+]?\d*\.?\d+)[,\s]+"
        r"loss_npo\(w\):\s*(?P<loss_npo>[-+]?\d*\.?\d+)[,\s]+"
        r"anti\(w\):\s*(?P<anti>[-+]?\d*\.?\d+)[,\s]+"
        r"kl\(w\):\s*(?P<kl>[-+]?\d*\.?\d+)[,\s]+"
        r"total:\s*(?P<total>[-+]?\d*\.?\d+)"
    )

    for line in text.splitlines():
        m = train_pattern.search(line)
        if m:
            row = {k: float(v) for k, v in m.groupdict().items() if k != "batch"}
            row["batch"] = int(m.group("batch"))
            train_rows.append(row)

    # -------------------------
    # 3) 解析评估指标名
    # -------------------------
    eval_metric_names = None
    eval_metric_line = re.search(r"\[Evaluation Metrics\]\s*\[(.*?)\]", text, re.S)
    if eval_metric_line:
        raw = eval_metric_line.group(1)
        eval_metric_names = [x.strip().strip("'").strip('"') for x in raw.split(",")]

    # 默认列名兜底
    if not eval_metric_names or len(eval_metric_names) != 7:
        eval_metric_names = [
            "SS Gender",
            "SS Profession",
            "SS Race",
            "SS Religion",
            "SS Score",
            "LM Score",
            "ICAT Score",
        ]

    # -------------------------
    # 4) 解析 [Scores]
    # -------------------------
    score_pattern = re.compile(r"^\[Scores\]\s+(.+)$")
    score_idx = 0
    for line in text.splitlines():
        m = score_pattern.match(line.strip())
        if m:
            values = re.split(r"[\t ]+", m.group(1).strip())
            values = [v for v in values if v != ""]
            if len(values) >= 7:
                vals = list(map(float, values[:7]))
                row = dict(zip(eval_metric_names, vals))
                row["eval_idx"] = score_idx
                eval_rows.append(row)
                score_idx += 1

    train_df = pd.DataFrame(train_rows).sort_values("batch").reset_index(drop=True)
    eval_df = pd.DataFrame(eval_rows).sort_values("eval_idx").reset_index(drop=True)

    # 如果有 eval_every，可以估算 eval 对应 step
    eval_every = None
    if "eval_every" in config:
        try:
            eval_every = int(float(config["eval_every"]))
        except Exception:
            eval_every = None

    if not eval_df.empty:
        if eval_every is not None:
            eval_df["eval_step"] = eval_df["eval_idx"] * eval_every
        else:
            eval_df["eval_step"] = eval_df["eval_idx"]

    return config, train_df, eval_df


def add_smoothed_column(df, col, window=5):
    smooth_col = f"{col}_smooth"
    df[smooth_col] = df[col].rolling(window=window, min_periods=1).mean()
    return smooth_col


def save_outputs(config, train_df, eval_df, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if not train_df.empty:
        train_df.to_csv(outdir / "train_metrics.csv", index=False, encoding="utf-8-sig")

    if not eval_df.empty:
        eval_df.to_csv(outdir / "eval_metrics.csv", index=False, encoding="utf-8-sig")


def plot_training_loss(train_df, outdir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["batch"], train_df["total"], label="total")
    plt.plot(train_df["batch"], train_df["loss_npo"], label="loss_npo")
    plt.plot(train_df["batch"], train_df["anti"], label="anti")
    plt.plot(train_df["batch"], train_df["kl"], label="kl")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "01_training_loss.png", dpi=200)
    plt.close()


def plot_unlearning_strength(train_df, outdir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["batch"], train_df["neg_log_ratio"], label="neg_log_ratio")
    plt.axhline(y=0, linestyle="--", linewidth=1)
    plt.axhline(y=4, linestyle="--", linewidth=1, label="ratio=4 reference")
    plt.xlabel("Batch")
    plt.ylabel("neg_log_ratio")
    plt.title("Unlearning Strength Trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "02_unlearning_strength.png", dpi=200)
    plt.close()


def plot_ster_lm(train_df, outdir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["batch"], train_df["ster_lm"], label="ster_lm")
    plt.plot(train_df["batch"], train_df["ster_ref_lm"], label="ster_ref_lm")
    plt.xlabel("Batch")
    plt.ylabel("LM Loss")
    plt.title("Stereotype LM vs Reference LM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "03_ster_lm_vs_ref.png", dpi=200)
    plt.close()


def plot_lr(train_df, outdir: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df["batch"], train_df["lr"], label="lr")
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "04_learning_rate.png", dpi=200)
    plt.close()


def plot_eval_metrics(eval_df, outdir: Path):
    x = eval_df["eval_step"] if "eval_step" in eval_df.columns else eval_df["eval_idx"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, eval_df["SS Score"], marker="o", label="SS Score")
    plt.plot(x, eval_df["LM Score"], marker="o", label="LM Score")
    plt.plot(x, eval_df["ICAT Score"], marker="o", label="ICAT Score")
    
    # 标记最佳 ICAT 点
    best_idx = eval_df["ICAT Score"].idxmax()
    best_x = x.iloc[best_idx]
    best_y = eval_df["ICAT Score"].iloc[best_idx]
    plt.scatter([best_x], [best_y], s=80, label="Best ICAT")
    plt.annotate(f"Best ICAT={best_y:.2f}", (best_x, best_y), fontsize=9)
    
    plt.xlabel("Evaluation Step")
    plt.ylabel("Score")
    plt.title("Evaluation Metric Trends")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "05_eval_metrics.png", dpi=200)
    plt.close()


def plot_bias_subscores(eval_df, outdir: Path):
    x = eval_df["eval_step"] if "eval_step" in eval_df.columns else eval_df["eval_idx"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, eval_df["SS Gender"], marker="o", label="SS Gender")
    plt.plot(x, eval_df["SS Profession"], marker="o", label="SS Profession")
    plt.plot(x, eval_df["SS Race"], marker="o", label="SS Race")
    plt.plot(x, eval_df["SS Religion"], marker="o", label="SS Religion")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Score")
    plt.title("Bias Subscores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "06_bias_subscores.png", dpi=200)
    plt.close()


def plot_tradeoff(eval_df, outdir: Path):
    plt.figure(figsize=(7, 6))
    plt.scatter(eval_df["SS Score"], eval_df["LM Score"])

    for _, row in eval_df.iterrows():
        label = int(row["eval_step"]) if "eval_step" in row else int(row["eval_idx"])
        plt.annotate(str(label), (row["SS Score"], row["LM Score"]), fontsize=8)

    plt.xlabel("SS Score")
    plt.ylabel("LM Score")
    plt.title("Bias-LM Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "07_tradeoff_scatter.png", dpi=200)
    plt.close()


def plot_overview_panel(train_df, eval_df, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # A: total / npo / anti / kl
    axes[0, 0].plot(train_df["batch"], train_df["total"], label="total")
    axes[0, 0].plot(train_df["batch"], train_df["loss_npo"], label="loss_npo")
    axes[0, 0].plot(train_df["batch"], train_df["anti"], label="anti")
    axes[0, 0].plot(train_df["batch"], train_df["kl"], label="kl")
    axes[0, 0].set_title("Loss Components")
    axes[0, 0].set_xlabel("Batch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # B: neg_log_ratio
    axes[0, 1].plot(train_df["batch"], train_df["neg_log_ratio"], label="neg_log_ratio")
    axes[0, 1].axhline(y=0, linestyle="--", linewidth=1)
    axes[0, 1].set_title("Unlearning Strength")
    axes[0, 1].set_xlabel("Batch")
    axes[0, 1].set_ylabel("neg_log_ratio")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # C: ster LM
    axes[1, 0].plot(train_df["batch"], train_df["ster_lm"], label="ster_lm")
    axes[1, 0].plot(train_df["batch"], train_df["ster_ref_lm"], label="ster_ref_lm")
    axes[1, 0].set_title("Stereotype LM")
    axes[1, 0].set_xlabel("Batch")
    axes[1, 0].set_ylabel("LM Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # D: eval metrics
    x = eval_df["eval_step"] if "eval_step" in eval_df.columns else eval_df["eval_idx"]
    axes[1, 1].plot(x, eval_df["SS Score"], marker="o", label="SS Score")
    axes[1, 1].plot(x, eval_df["LM Score"], marker="o", label="LM Score")
    axes[1, 1].plot(x, eval_df["ICAT Score"], marker="o", label="ICAT Score")
    axes[1, 1].set_title("Evaluation Metrics")
    axes[1, 1].set_xlabel("Evaluation Step")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Bias Unlearning Training Overview", fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / "08_overview_panel.png", dpi=220)
    plt.close()


def plot_with_smoothing(train_df, eval_df, outdir: Path):
    # 可选：给关键曲线做平滑版
    tmp_train = train_df.copy()
    tmp_eval = eval_df.copy()

    add_smoothed_column(tmp_train, "total", window=5)
    add_smoothed_column(tmp_train, "neg_log_ratio", window=5)
    add_smoothed_column(tmp_train, "ster_lm", window=5)
    add_smoothed_column(tmp_train, "ster_ref_lm", window=5)

    plt.figure(figsize=(10, 6))
    plt.plot(tmp_train["batch"], tmp_train["total"], alpha=0.35, label="total(raw)")
    plt.plot(tmp_train["batch"], tmp_train["total_smooth"], linewidth=2, label="total(smooth)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Smoothed Total Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "09_total_loss_smoothed.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(tmp_train["batch"], tmp_train["neg_log_ratio"], alpha=0.35, label="neg_log_ratio(raw)")
    plt.plot(tmp_train["batch"], tmp_train["neg_log_ratio_smooth"], linewidth=2, label="neg_log_ratio(smooth)")
    plt.xlabel("Batch")
    plt.ylabel("neg_log_ratio")
    plt.title("Smoothed Unlearning Strength")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "10_neg_ratio_smoothed.png", dpi=200)
    plt.close()

    if not tmp_eval.empty:
        add_smoothed_column(tmp_eval, "SS Score", window=3)
        add_smoothed_column(tmp_eval, "LM Score", window=3)
        add_smoothed_column(tmp_eval, "ICAT Score", window=3)
        x = tmp_eval["eval_step"] if "eval_step" in tmp_eval.columns else tmp_eval["eval_idx"]

        plt.figure(figsize=(10, 6))
        plt.plot(x, tmp_eval["SS Score"], alpha=0.35, marker="o", label="SS(raw)")
        plt.plot(x, tmp_eval["SS Score_smooth"], linewidth=2, label="SS(smooth)")
        plt.plot(x, tmp_eval["LM Score"], alpha=0.35, marker="o", label="LM(raw)")
        plt.plot(x, tmp_eval["LM Score_smooth"], linewidth=2, label="LM(smooth)")
        plt.plot(x, tmp_eval["ICAT Score"], alpha=0.35, marker="o", label="ICAT(raw)")
        plt.plot(x, tmp_eval["ICAT Score_smooth"], linewidth=2, label="ICAT(smooth)")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Score")
        plt.title("Smoothed Evaluation Trends")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "11_eval_metrics_smoothed.png", dpi=200)
        plt.close()


def print_summary(config, train_df, eval_df):
    print("\n===== Parsed Summary =====")
    print(f"Config keys: {len(config)}")
    print(f"Training rows: {len(train_df)}")
    print(f"Eval rows: {len(eval_df)}")

    if not train_df.empty:
        print(f"Batch range: {train_df['batch'].min()} -> {train_df['batch'].max()}")
        print(f"Final total loss: {train_df['total'].iloc[-1]:.4f}")
        print(f"Final neg_log_ratio: {train_df['neg_log_ratio'].iloc[-1]:.4f}")

    if not eval_df.empty:
        best_icat_idx = eval_df["ICAT Score"].idxmax()
        best_row = eval_df.loc[best_icat_idx]
        print(
            f"Best ICAT: {best_row['ICAT Score']:.2f} "
            f"at eval_step={best_row['eval_step']}"
        )
        print(
            f"At best ICAT -> SS={best_row['SS Score']:.2f}, "
            f"LM={best_row['LM Score']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Parse simplified bias-unlearning report and plot trends.")
    parser.add_argument("--input", type=str, required=True, help="Path to the report text file.")
    parser.add_argument("--output_dir", type=str, default="report_plots", help="Directory to save figures.")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.output_dir)

    text = input_path.read_text(encoding="utf-8")
    config, train_df, eval_df = parse_report(text)

    if train_df.empty:
        raise ValueError("No training rows parsed. Please check the log format.")

    if eval_df.empty:
        print("Warning: No evaluation rows parsed.")

    save_outputs(config, train_df, eval_df, outdir)

    plot_training_loss(train_df, outdir)
    plot_unlearning_strength(train_df, outdir)
    plot_ster_lm(train_df, outdir)
    plot_lr(train_df, outdir)

    if not eval_df.empty:
        plot_eval_metrics(eval_df, outdir)
        plot_bias_subscores(eval_df, outdir)
        plot_tradeoff(eval_df, outdir)

    if not eval_df.empty:
        plot_overview_panel(train_df, eval_df, outdir)

    plot_with_smoothing(train_df, eval_df, outdir)

    print_summary(config, train_df, eval_df)
    print(f"\nAll outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()