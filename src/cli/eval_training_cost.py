# src/cli/eval_training_cost.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from src.cli.common import load_dataset
from src.cli.model_factory import load_twopass_eval_models
from src.models.labeler import PseudoLabeler
from src.utils.lkh_solver import default_lkh_executable


TRAIN_RE = re.compile(
    r"^\[train\]\s+epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+L_token=(?P<l_token>[-+0-9.eE]+)\s+L_bc=(?P<l_bc>[-+0-9.eE]+)\s+"
    r"\((?P<parts>.*?)\)\s+.*time=(?P<elapsed>[-+0-9.eE]+)s$"
)
VAL_PAIR_RE = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-_.")
    return cleaned or "run"


def build_run_tag(*, ckpt_path: str) -> str:
    ckpt_obj = Path(ckpt_path)
    stem = ckpt_obj.parent.name if ckpt_obj.parent.name else ckpt_obj.stem
    return sanitize_tag(f"{utc_timestamp()}_{stem}_training_cost")


def auto_find_train_log(*, ckpt_path: str, log_dir: str = "checkpoints") -> Optional[str]:
    ckpt_obj = Path(ckpt_path).resolve()
    run_dir = ckpt_obj.parent.name
    sibling = ckpt_obj.parent / "train.log"
    if sibling.exists():
        return str(sibling)

    candidates = sorted(Path(log_dir).glob("*.log"))
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if run_dir and run_dir in text:
            return str(path)
    return None


def parse_train_log(log_path: str) -> Dict[str, Any]:
    lines = Path(log_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    env_device = None
    ckpt_dir = None
    train_points: List[Dict[str, Any]] = []
    val_points: List[Dict[str, Any]] = []
    last_step = None
    last_epoch = None
    done_seen = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[env] device="):
            env_device = line.split("=", 1)[1].strip()
            continue
        if line.startswith("[ckpt] Checkpoints will be saved to:"):
            ckpt_dir = line.split(":", 1)[1].strip()
            continue
        if line.startswith("[done]"):
            done_seen = True
            continue

        m_train = TRAIN_RE.match(line)
        if m_train:
            parts_text = str(m_train.group("parts"))
            parts = {
                k: float(v)
                for k, v in VAL_PAIR_RE.findall(parts_text)
            }
            point = {
                "epoch": int(m_train.group("epoch")),
                "step": int(m_train.group("step")),
                "loss": float(m_train.group("loss")),
                "L_token": float(m_train.group("l_token")),
                "L_bc": float(m_train.group("l_bc")),
                "elapsed_sec": float(m_train.group("elapsed")),
                "parts": parts,
            }
            train_points.append(point)
            last_step = point["step"]
            last_epoch = point["epoch"]
            continue

        if line.startswith("[val] "):
            metrics = {k: float(v) for k, v in VAL_PAIR_RE.findall(line)}
            if metrics:
                val_points.append(
                    {
                        "step": int(last_step) if last_step is not None else None,
                        "epoch": int(last_epoch) if last_epoch is not None else None,
                        "metrics": metrics,
                    }
                )

    return {
        "log_path": str(Path(log_path).resolve()),
        "env_device": env_device,
        "ckpt_dir": ckpt_dir,
        "train_points": train_points,
        "val_points": val_points,
        "done_seen": done_seen,
    }


def infer_training_parallelism(*, env_device: Optional[str], num_gpus_override: Optional[int]) -> Dict[str, Any]:
    if num_gpus_override is not None:
        num_gpus = max(0, int(num_gpus_override))
    else:
        device_text = str(env_device or "").lower()
        if device_text.startswith("cuda"):
            num_gpus = 1
        else:
            num_gpus = 0

    if num_gpus <= 0:
        mode = "cpu_or_unknown"
    elif num_gpus == 1:
        mode = "single_gpu"
    else:
        mode = "multi_gpu"
    return {
        "num_gpus": int(num_gpus),
        "parallelism_mode": mode,
    }


def build_param_report(*, ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    r = int(ckpt_args.get("r", 4))
    model_bundle = load_twopass_eval_models(
        ckpt_path=str(ckpt_path),
        device=torch.device("cpu"),
        r=r,
    )

    modules = {
        "leaf_encoder": model_bundle.leaf_encoder,
        "merge_encoder": model_bundle.merge_encoder,
        "decoder": model_bundle.decoder,
    }
    per_module = {
        name: int(sum(p.numel() for p in module.parameters()))
        for name, module in modules.items()
    }
    total = int(sum(per_module.values()))
    trainable = int(sum(p.numel() for module in modules.values() for p in module.parameters() if p.requires_grad))
    return {
        "d_model": int(model_bundle.d_model),
        "state_mode": str(model_bundle.state_mode),
        "matching_max_used": int(model_bundle.matching_max_used),
        "per_module": per_module,
        "total_params": total,
        "trainable_params": trainable,
    }


def write_curve_csv(*, path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def plot_convergence(*, train_points: List[Dict[str, Any]], val_points: List[Dict[str, Any]], output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Failed to import matplotlib, skipping convergence plot: {exc}")
        return False

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if train_points:
        steps = [p["step"] for p in train_points]
        axes[0].plot(steps, [p["loss"] for p in train_points], label="Train loss", color="#1f77b4", linewidth=1.6)
        axes[0].plot(steps, [p["L_token"] for p in train_points], label="Train token", color="#ff7f0e", linewidth=1.0, alpha=0.8)
        axes[0].plot(steps, [p["L_bc"] for p in train_points], label="Train BC", color="#2ca02c", linewidth=1.0, alpha=0.8)
        axes[0].set_ylabel("Train Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

    if val_points:
        filtered_val = [
            p for p in val_points
            if p.get("step") is not None and p.get("metrics", {}).get("val_loss") is not None
        ]
        val_steps = [p["step"] for p in filtered_val]
        val_loss = [float(p["metrics"]["val_loss"]) for p in filtered_val]
        val_gap = [p["metrics"].get("val_decode_gap_mean") for p in filtered_val]
        if val_steps:
            axes[1].plot(val_steps, val_loss, label="Val loss", color="#d62728", linewidth=1.6)
        if val_steps and any(v is not None for v in val_gap):
            axes2 = axes[1].twinx()
            axes2.plot(
                val_steps,
                [float(v) * 100.0 if v is not None else float("nan") for v in val_gap],
                label="Val gap (%)",
                color="#9467bd",
                linewidth=1.2,
                alpha=0.8,
            )
            axes2.set_ylabel("Val Gap (%)")
            axes2.legend(loc="upper right")
        axes[1].set_ylabel("Val Loss")
        axes[1].set_xlabel("Global Step")
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.25)

    fig.suptitle("2-Pass Training Convergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def load_teacher_dataset(path: str) -> Any:
    dataset = load_dataset(path)
    if len(dataset) > 0:
        sample0 = dataset[0]
        if hasattr(sample0, "pos") and hasattr(sample0, "spanner_edge_index"):
            return dataset
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported teacher dataset format at {path}")


def _teacher_timing_worker(args_tuple):
    idx, data_payload, labeler = args_tuple
    t0 = time.perf_counter()
    try:
        labeler.extract_teacher_supervision(data_payload)
        return idx, True, time.perf_counter() - t0, ""
    except Exception as exc:
        return idx, False, time.perf_counter() - t0, str(exc)


def time_teacher_generation(
    *,
    data_pt: str,
    labeler: PseudoLabeler,
    sample_idx: int,
    sample_idx_end: Optional[int],
    num_workers: int,
) -> Dict[str, Any]:
    dataset = load_teacher_dataset(data_pt)
    total = len(dataset)
    start = max(0, int(sample_idx))
    end = total if sample_idx_end is None else min(int(sample_idx_end), total)
    indices = list(range(start, end))

    sample_durations: List[float] = []
    failures: List[Dict[str, Any]] = []
    wall_t0 = time.perf_counter()

    if int(num_workers) <= 0:
        pbar = tqdm(indices, desc="[teacher] timing", total=len(indices))
        try:
            for idx in pbar:
                t0 = time.perf_counter()
                try:
                    labeler.extract_teacher_supervision(dataset[idx])
                except Exception as exc:
                    failures.append({"sample_idx": int(idx), "reason": str(exc)})
                sample_durations.append(time.perf_counter() - t0)
        finally:
            pbar.close()
    else:
        max_inflight = max(int(num_workers) * 4, 32)

        def submit_one(executor, cur_idx: int):
            payload = labeler.simplify_data_for_ipc(dataset[cur_idx])
            return executor.submit(_teacher_timing_worker, (cur_idx, payload, labeler))

        next_pos = 0
        pbar = tqdm(total=len(indices), desc="[teacher] parallel timing")
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            in_flight = set()
            try:
                while next_pos < len(indices) and len(in_flight) < max_inflight:
                    in_flight.add(submit_one(executor, indices[next_pos]))
                    next_pos += 1
                while in_flight:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        idx, ok, dt, reason = future.result()
                        sample_durations.append(float(dt))
                        if not ok:
                            failures.append({"sample_idx": int(idx), "reason": reason})
                        pbar.update(1)
                        if next_pos < len(indices):
                            in_flight.add(submit_one(executor, indices[next_pos]))
                            next_pos += 1
            finally:
                pbar.close()

    wall_sec = time.perf_counter() - wall_t0
    num_samples = len(indices)
    success_count = int(num_samples - len(failures))
    throughput = float(success_count / wall_sec) if wall_sec > 1e-12 else 0.0
    avg_sample_sec = float(sum(sample_durations) / max(len(sample_durations), 1))
    dataset_fraction = float(num_samples / max(total, 1))
    projected_full_sec = float(wall_sec / dataset_fraction) if dataset_fraction > 1e-12 else wall_sec

    return {
        "data_pt": str(Path(data_pt).resolve()),
        "dataset_total_samples": int(total),
        "timed_sample_idx": int(start),
        "timed_sample_idx_end": int(end),
        "timed_num_samples": int(num_samples),
        "num_workers": int(num_workers),
        "wall_clock_sec": float(wall_sec),
        "throughput_samples_per_sec": throughput,
        "avg_sample_sec": avg_sample_sec,
        "projected_full_dataset_sec": projected_full_sec,
        "projected_full_dataset_hours": float(projected_full_sec / 3600.0),
        "success_count": int(success_count),
        "failure_count": int(len(failures)),
        "failures_preview": failures[:10],
    }


def build_report_lines(*, summary: Dict[str, Any]) -> List[str]:
    teacher = summary.get("training_data_generation", summary.get("teacher_generation", {}))
    train = summary.get("training_time", {})
    params = summary.get("parameters", {})
    curves = summary.get("curves", {})
    artifacts = summary.get("artifacts", {})

    lines = ["", "=" * 78, "Training Cost Report", "=" * 78]
    if teacher:
        lines.append("[Training Data Generation]")
        lines.append(
            f"  timed {teacher['timed_num_samples']}/{teacher['dataset_total_samples']} samples "
            f"in {teacher['wall_clock_sec']:.2f}s "
            f"({teacher['throughput_samples_per_sec']:.2f} samples/s)"
        )
        lines.append(
            f"  projected full dataset: {teacher['projected_full_dataset_sec']:.2f}s "
            f"({teacher['projected_full_dataset_hours']:.3f} h)"
        )
        lines.append(
            f"  workers={teacher['num_workers']} success={teacher['success_count']} fail={teacher['failure_count']}"
        )
    if train:
        lines.append("[Training Time]")
        lines.append(
            f"  mode={train.get('parallelism_mode')} num_gpus={train.get('num_gpus')} "
            f"wall_clock={train.get('wall_clock_sec', 0.0):.2f}s "
            f"gpu_hours={train.get('gpu_hours', 0.0):.3f}"
        )
        lines.append(
            f"  epochs={train.get('max_epoch_plus_one')} steps={train.get('max_step')} "
            f"log={train.get('log_path', 'N/A')}"
        )
    if params:
        lines.append("[Parameters]")
        lines.append(f"  total={params.get('total_params', 0):,} trainable={params.get('trainable_params', 0):,}")
        for name, count in params.get("per_module", {}).items():
            lines.append(f"  {name}: {int(count):,}")
    if curves:
        lines.append("[Convergence]")
        lines.append(
            f"  train_points={curves.get('num_train_points', 0)} "
            f"val_points={curves.get('num_val_points', 0)} "
            f"best_val_loss={curves.get('best_val_loss')} at step={curves.get('best_val_step')}"
        )
    if artifacts:
        lines.append("[Artifacts]")
        for key, value in artifacts.items():
            if value:
                lines.append(f"  {key}: {value}")
    lines.append("=" * 78)
    return lines


def print_report(*, summary: Dict[str, Any]) -> None:
    for line in build_report_lines(summary=summary):
        print(line)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Training cost report for the 2-pass model.")
    parser.add_argument("--ckpt", type=str, required=True, help="path to a 2-pass checkpoint")
    parser.add_argument("--train_log", type=str, default=None, help="path to the human-readable training log; auto-detected when omitted")
    parser.add_argument("--log_dir", type=str, default="checkpoints", help="directory searched when auto-detecting train logs")
    parser.add_argument("--teacher_data_pt", type=str, default=None, help="preprocessed training dataset used for teacher generation timing")
    parser.add_argument("--teacher_sample_idx", type=int, default=0, help="teacher timing start sample index")
    parser.add_argument("--teacher_sample_idx_end", type=int, default=None, help="teacher timing end sample index (exclusive); omit to time the full dataset")
    parser.add_argument("--teacher_num_workers", type=int, default=None, help="workers for teacher timing; defaults to the checkpoint training num_workers")
    parser.add_argument("--teacher_lkh_runs", type=int, default=None, help="override teacher LKH runs; default comes from checkpoint args")
    parser.add_argument("--teacher_lkh_timeout", type=float, default=None, help="override teacher LKH timeout in seconds")
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable(), help="path to LKH executable for teacher timing")
    parser.add_argument("--num_gpus_override", type=int, default=None, help="override the GPU count used for GPU-hour calculation")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_training_cost", help="directory for report artifacts")
    parser.add_argument("--run_tag", type=str, default=None, help="optional output file prefix")
    parser.add_argument("--skip_teacher_timing", action="store_true", help="skip teacher generation timing")
    parser.add_argument("--skip_curve_plot", action="store_true", help="skip convergence plot generation")
    args = parser.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = str(args.run_tag or build_run_tag(ckpt_path=str(args.ckpt)))

    log_path = args.train_log or auto_find_train_log(ckpt_path=str(args.ckpt), log_dir=str(args.log_dir))
    log_info = parse_train_log(log_path) if log_path else None

    param_report = build_param_report(ckpt_path=str(args.ckpt))

    curves_summary: Dict[str, Any] = {}
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    if log_info is not None:
        train_rows = [
            {
                "epoch": p["epoch"],
                "step": p["step"],
                "elapsed_sec": p["elapsed_sec"],
                "loss": p["loss"],
                "L_token": p["L_token"],
                "L_bc": p["L_bc"],
                "loss_cross": p["parts"].get("loss_cross"),
                "loss_iface": p["parts"].get("loss_iface"),
                "loss_total_part": p["parts"].get("loss_total"),
            }
            for p in log_info["train_points"]
        ]
        val_rows = []
        for p in log_info["val_points"]:
            row = {
                "epoch": p.get("epoch"),
                "step": p.get("step"),
            }
            row.update(p["metrics"])
            val_rows.append(row)

        if train_rows:
            max_elapsed = max(float(p["elapsed_sec"]) for p in train_rows)
            parallel = infer_training_parallelism(
                env_device=log_info.get("env_device"),
                num_gpus_override=args.num_gpus_override,
            )
            training_time = {
                "log_path": log_info["log_path"],
                "env_device": log_info.get("env_device"),
                "parallelism_mode": parallel["parallelism_mode"],
                "num_gpus": parallel["num_gpus"],
                "wall_clock_sec": float(max_elapsed),
                "wall_clock_hours": float(max_elapsed / 3600.0),
                "gpu_hours": float(max_elapsed / 3600.0 * parallel["num_gpus"]),
                "max_step": int(max(p["step"] for p in train_rows)),
                "max_epoch_plus_one": int(max(p["epoch"] for p in train_rows) + 1),
                "done_seen": bool(log_info.get("done_seen")),
            }
        else:
            training_time = {}

        if val_rows:
            best_row = min(
                (row for row in val_rows if row.get("val_loss") is not None),
                key=lambda row: float(row["val_loss"]),
            )
            curves_summary = {
                "num_train_points": len(train_rows),
                "num_val_points": len(val_rows),
                "best_val_loss": float(best_row["val_loss"]),
                "best_val_step": int(best_row["step"]) if best_row.get("step") is not None else None,
                "best_val_epoch": int(best_row["epoch"]) if best_row.get("epoch") is not None else None,
            }
        else:
            curves_summary = {
                "num_train_points": len(train_rows),
                "num_val_points": 0,
                "best_val_loss": None,
                "best_val_step": None,
                "best_val_epoch": None,
            }
    else:
        training_time = {}

    train_csv = output_dir / f"{run_tag}_train_curve.csv"
    val_csv = output_dir / f"{run_tag}_val_curve.csv"
    if train_rows:
        write_curve_csv(
            path=train_csv,
            rows=train_rows,
            fieldnames=["epoch", "step", "elapsed_sec", "loss", "L_token", "L_bc", "loss_cross", "loss_iface", "loss_total_part"],
        )
    if val_rows:
        val_fieldnames = ["epoch", "step"] + sorted({k for row in val_rows for k in row.keys() if k not in {"epoch", "step"}})
        write_curve_csv(path=val_csv, rows=val_rows, fieldnames=val_fieldnames)

    curve_png = output_dir / f"{run_tag}_convergence.png"
    curve_plot_written = False
    if (not args.skip_curve_plot) and log_info is not None and (train_rows or val_rows):
        curve_plot_written = plot_convergence(
            train_points=log_info["train_points"],
            val_points=log_info["val_points"],
            output_path=curve_png,
        )

    teacher_report = {}
    if not args.skip_teacher_timing:
        teacher_data_pt = args.teacher_data_pt or ckpt_args.get("train_pt")
        if teacher_data_pt:
            teacher_num_workers = (
                int(args.teacher_num_workers)
                if args.teacher_num_workers is not None
                else int(ckpt_args.get("num_workers", 0))
            )
            labeler = PseudoLabeler(
                two_opt_passes=int(ckpt_args.get("two_opt_passes", 30)),
                use_lkh=bool(ckpt_args.get("use_lkh", False)),
                lkh_exe=str(args.lkh_exe),
                prefer_cpu=True,
                teacher_mode="spanner_lkh",
                teacher_lkh_runs=int(
                    args.teacher_lkh_runs
                    if args.teacher_lkh_runs is not None
                    else ckpt_args.get("teacher_lkh_runs", 1)
                ),
                teacher_lkh_timeout=(
                    None
                    if args.teacher_lkh_timeout is None
                    else (None if float(args.teacher_lkh_timeout) <= 0 else float(args.teacher_lkh_timeout))
                ),
            )
            teacher_report = time_teacher_generation(
                data_pt=str(teacher_data_pt),
                labeler=labeler,
                sample_idx=int(args.teacher_sample_idx),
                sample_idx_end=args.teacher_sample_idx_end,
                num_workers=teacher_num_workers,
            )

    summary = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "ckpt": str(Path(args.ckpt).resolve()),
            "train_log": (None if log_path is None else str(Path(log_path).resolve())),
            "teacher_data_pt": (
                None
                if (args.teacher_data_pt or ckpt_args.get("train_pt")) is None
                else str(Path(args.teacher_data_pt or ckpt_args.get("train_pt")).resolve())
            ),
        },
        "parameters": param_report,
        "training_time": training_time,
        "training_data_generation": teacher_report,
        "teacher_generation": teacher_report,
        "curves": curves_summary,
        "artifacts": {
            "train_curve_csv": str(train_csv.resolve()) if train_rows else None,
            "val_curve_csv": str(val_csv.resolve()) if val_rows else None,
            "convergence_png": str(curve_png.resolve()) if curve_plot_written and curve_png.exists() else None,
        },
    }

    summary_path = output_dir / f"{run_tag}_summary.json"
    report_txt = output_dir / f"{run_tag}_report.txt"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_txt.write_text("\n".join(build_report_lines(summary=summary)) + "\n", encoding="utf-8")
    print_report(summary=summary)
    print(f"[save] summary JSON   -> {summary_path}")
    print(f"[save] text report    -> {report_txt}")
    if train_rows:
        print(f"[save] train curve CSV -> {train_csv}")
    if val_rows:
        print(f"[save] val curve CSV   -> {val_csv}")
    if curve_plot_written and curve_png.exists():
        print(f"[save] convergence PNG -> {curve_png}")


if __name__ == "__main__":
    main()
