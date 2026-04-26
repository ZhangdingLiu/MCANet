import argparse
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt


MODEL_RUNS = OrderedDict({
    "vgg16_nocsra": {
        "run_dir": "logs/rescuenet_vgg16_nocsra_1_2026-04-04_01-16-50",
        "log_file": "vgg16_nocsra_rescuenet_log.txt",
        "queue_log": "logs_gpu0_v2.txt",
        "start_name": "vgg16_nocsra",
    },
    "efficientnet_b4_nocsra": {
        "run_dir": "logs/rescuenet_efficientnet_b4_nocsra_1_2026-04-04_00-31-34",
        "log_file": "efficientnet_b4_nocsra_rescuenet_log.txt",
        "queue_log": "logs_gpu1_v2.txt",
        "start_name": "efficientnet_b4_nocsra",
    },
    "VIT_B16_448_BASE": {
        "run_dir": "logs/rescuenet_VIT_B16_448_BASE_1_2026-04-03_17-05-07",
        "log_file": "VIT_B16_448_BASE_rescuenet_log.txt",
        "queue_log": "logs_gpu0_v2.txt",
        "start_name": "VIT_B16_448_BASE",
    },
    "mobilenet_nocsra": {
        "run_dir": "logs/rescuenet_mobilenet_nocsra_1_2026-04-04_02-28-08",
        "log_file": "mobilenet_nocsra_rescuenet_log.txt",
        "queue_log": "logs_gpu0_v2.txt",
        "start_name": "mobilenet_nocsra",
    },
    "resnet101_nocsra": {
        "run_dir": "logs/rescuenet_resnet101_nocsra_1_2026-04-04_01-43-50",
        "log_file": "resnet101_nocsra_rescuenet_log.txt",
        "queue_log": "logs_gpu1_v2.txt",
        "start_name": "resnet101_nocsra",
    },
    "res2net101_nocsra": {
        "run_dir": "logs/rescuenet_res2net101_nocsra_1_2026-04-03_17-05-09",
        "log_file": "res2net101_nocsra_rescuenet_log.txt",
        "queue_log": "logs_gpu1_v2.txt",
        "start_name": "res2net101_nocsra",
    },
    "res2net101_csra": {
        "run_dir": "logs/rescuenet_res2net101_csra_1_2026-04-03_19-08-48",
        "log_file": "res2net101_csra_rescuenet_log.txt",
        "queue_log": "logs_gpu0_v2.txt",
        "start_name": "res2net101_csra h=1",
    },
})


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training diagnostics for Table 3 models.")
    parser.add_argument("--output_dir", default="figures", help="Directory for generated PNGs and summary.")
    return parser.parse_args()


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_queue_segment(queue_text, start_name):
    start_marker = f"Starting: {start_name}"
    start_idx = queue_text.find(start_marker)
    if start_idx == -1:
        raise ValueError(f"Could not find queue start marker for {start_name}")

    done_marker = f"Done: {start_name}"
    done_idx = queue_text.find(done_marker, start_idx)
    if done_idx == -1:
        done_idx = len(queue_text)

    return queue_text[start_idx:done_idx]


def parse_train_metrics(segment):
    train_losses = []
    lrs = OrderedDict()

    epoch_line = re.compile(r"Epoch (\d+)\[\d+/\d+\]: loss:[0-9.]+, lr:([0-9.]+), time:[0-9.]+")
    avg_line = re.compile(r"Average training loss: ([0-9.]+)")

    for match in epoch_line.finditer(segment):
        epoch = int(match.group(1))
        lr = float(match.group(2))
        if epoch not in lrs:
            lrs[epoch] = lr

    for match in avg_line.finditer(segment):
        train_losses.append(float(match.group(1)))

    lr_values = [lrs[idx] for idx in sorted(lrs.keys())]
    return train_losses, lr_values


def parse_val_map(log_path):
    text = read_text(log_path)
    matches = re.findall(r"mAP: ([0-9.]+)", text)
    return [float(value) for value in matches]


def classify_curve(train_losses, val_maps, lrs):
    best_epoch = max(range(len(val_maps)), key=lambda idx: val_maps[idx]) + 1
    best_val = val_maps[best_epoch - 1]
    last_val = val_maps[-1]
    last5 = val_maps[-5:]
    tail_span = max(last5) - min(last5)
    lr_drops = sum(1 for i in range(1, len(lrs)) if lrs[i] < lrs[i - 1] - 1e-9)
    val_diffs = [val_maps[i] - val_maps[i - 1] for i in range(1, len(val_maps))]
    large_swings = sum(1 for diff in val_diffs if abs(diff) >= 1.0)

    if large_swings >= 6:
        status = "UNSTABLE"
    elif best_epoch <= 24 and best_val - last_val >= 0.8 and tail_span <= 0.6:
        status = "OVERFIT"
    elif best_epoch >= 28 and (best_val - max(val_maps[:-3]) > 0.2 or last_val >= best_val - 0.1):
        status = "STILL_IMPROVING"
    else:
        status = "CONVERGED"

    lr_working = "YES" if lr_drops > 0 else "NO"
    overfit = "YES" if status == "OVERFIT" else "NO"
    unstable = "YES" if status == "UNSTABLE" else "NO"
    convergence = "YES" if status == "CONVERGED" else "NO"

    assessment = {
        "status": status,
        "best_epoch": best_epoch,
        "best_val_map": best_val,
        "last_val_map": last_val,
        "tail_span": tail_span,
        "lr_drops": lr_drops,
        "converged_by_30": convergence,
        "overfitting_sign": overfit,
        "lr_schedule_working": lr_working,
        "instability_sign": unstable,
    }
    return assessment


def plot_model(model_name, train_losses, val_maps, lrs, assessment, output_dir):
    epochs = list(range(1, len(train_losses) + 1))
    best_epoch = assessment["best_epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(epochs, train_losses, color="#d55e00", linewidth=2)
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title(f"{model_name} Training Diagnostics")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_maps, color="#0072b2", linewidth=2)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[1].scatter([best_epoch], [val_maps[best_epoch - 1]], color="#0072b2", zorder=3)
    axes[1].set_ylabel("Val mAP")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, lrs, color="#009e73", linewidth=2)
    axes[2].axvline(best_epoch, color="gray", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, f"training_curves_{model_name}.png")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def build_summary_line(model_name, assessment):
    return (
        f"| {model_name} | {assessment['status']} | {assessment['best_epoch']} | "
        f"{assessment['best_val_map']:.2f} | {assessment['last_val_map']:.2f} | "
        f"{assessment['converged_by_30']} | {assessment['overfitting_sign']} | "
        f"{assessment['lr_schedule_working']} | {assessment['instability_sign']} |"
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    queue_cache = {}
    summary_lines = [
        "# Training Diagnostics Summary",
        "",
        "| Model | Status | Best Epoch | Best Val mAP | Epoch 30 Val mAP | Converged by 30 | Overfit | LR Drops Observed | Unstable |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]

    for model_name, config in MODEL_RUNS.items():
        queue_path = config["queue_log"]
        if queue_path not in queue_cache:
            queue_cache[queue_path] = read_text(queue_path)

        segment = get_queue_segment(queue_cache[queue_path], config["start_name"])
        train_losses, lrs = parse_train_metrics(segment)
        val_maps = parse_val_map(os.path.join(config["run_dir"], config["log_file"]))

        if not (len(train_losses) == len(val_maps) == len(lrs) == 30):
            raise ValueError(
                f"{model_name}: expected 30 points for all series, got "
                f"train={len(train_losses)} val={len(val_maps)} lr={len(lrs)}"
            )

        assessment = classify_curve(train_losses, val_maps, lrs)
        plot_model(model_name, train_losses, val_maps, lrs, assessment, args.output_dir)
        summary_lines.append(build_summary_line(model_name, assessment))

    summary_path = os.path.join(args.output_dir, "training_diagnostics_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
