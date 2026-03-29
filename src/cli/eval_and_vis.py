# src/cli/eval_and_vis.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch

from src.cli.eval_settings import describe_settings, resolve_eval_settings


AVAILABLE_SETTINGS = (
    "greedy",
    "exact",
    "guided_lkh",
    "pure_lkh",
    "pomo",
    "neurolkh",
)

DEFAULT_SETTINGS = (
    "greedy",
    "guided_lkh",
    "pure_lkh",
    "pomo",
    "neurolkh",
)

SETTING_GROUPS = {
    "default": DEFAULT_SETTINGS,
    "all": ("greedy", "exact", "guided_lkh", "pure_lkh", "pomo", "neurolkh"),
    "ours": ("greedy", "exact", "guided_lkh"),
    "baselines": ("pomo", "neurolkh"),
    "reference": ("pure_lkh",),
    "lkh": ("guided_lkh", "pure_lkh"),
}

SETTING_ALIASES = {
    "guided": "guided_lkh",
    "guidedlkh": "guided_lkh",
    "pure": "pure_lkh",
    "lkh_pure": "pure_lkh",
    "ex": "exact",
}

SETTING_DISPLAY_NAMES = {
    "greedy": "Greedy",
    "exact": "Exact Sparse",
    "guided_lkh": "Guided LKH",
    "pure_lkh": "Pure LKH",
    "pomo": "POMO",
    "neurolkh": "NeuroLKH",
}

SETTING_COLORS = {
    "greedy": "red",
    "exact": "darkorange",
    "guided_lkh": "blue",
    "pure_lkh": "green",
    "pomo": "purple",
    "neurolkh": "brown",
}


def parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    skip = {"num_faces"}
    for k in dir(data):
        if k.startswith("_") or k in skip:
            continue
        v = getattr(data, k, None)
        if isinstance(v, torch.Tensor):
            try:
                setattr(data, k, v.to(device))
            except Exception:
                pass
    return data


def load_dataset(path: str):
    from src.dataprep.dataset import smart_load_dataset

    return smart_load_dataset(path)


def plot_tour(pos, edge_index, ax, color="blue", lw=1.5, alpha=1.0, label=None, linestyle="-"):
    pos_np = pos.detach().cpu().numpy()
    edge_index_np = edge_index.detach().cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        u, v = edge_index_np[0, i], edge_index_np[1, i]
        ax.plot(
            [pos_np[u, 0], pos_np[v, 0]],
            [pos_np[u, 1], pos_np[v, 1]],
            color=color,
            linewidth=lw,
            alpha=alpha,
            linestyle=linestyle,
            label=label if i == 0 else None,
        )


def prepare_edges(order, device: torch.device) -> torch.Tensor:
    model_order = torch.as_tensor(order, device=device, dtype=torch.long)
    nxt = torch.roll(model_order, shifts=-1, dims=0)
    return torch.stack([model_order, nxt], dim=0)


def placeholder_result() -> SimpleNamespace:
    return SimpleNamespace(feasible=False, length=float("inf"), duration=0.0, order=[])


def log_progress(prefix: str, message: str) -> None:
    print(f"{prefix} {message}", flush=True)


def load_pomo(device: torch.device, ckpt_path: str):
    from src_baselines.pomo.POMO.TSPEnv import TSPEnv
    from src_baselines.pomo.POMO.TSPModel import TSPModel

    class CustomTSPEnv(TSPEnv):
        def load_problems(self, batch_size, aug_factor=1):
            pass

        def set_problems(self, problems):
            self.batch_size = problems.size(0)
            self.problem_size = problems.size(1)
            self.problems = problems
            self.device = problems.device
            self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[:, None].expand(self.batch_size, self.pomo_size)
            self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None, :].expand(self.batch_size, self.pomo_size)

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    model = TSPModel(**model_params).to(device)
    pomo_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(pomo_ckpt["model_state_dict"])
    model.eval()
    env = CustomTSPEnv(problem_size=100, pomo_size=100)
    env.device = device
    return model, env


def load_neurolkh(device: torch.device, ckpt_path: str):
    from src_baselines.neurolkh.net.sgcn_model import SparseGCNModel

    model = SparseGCNModel(problem="tsp").to(device)
    nlkh_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(nlkh_ckpt["model"])
    model.eval()
    return model


class TeacherDataset(torch.utils.data.Dataset):
    def __init__(self, tasks, labeler):
        self.tasks = tasks
        self.labeler = labeler

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        data, ts = self.tasks[idx]
        teacher_lab = self.labeler.label_one(data=data, tokens_slice=ts, device=torch.device("cpu"), eid_offset=0)
        return teacher_lab.stats["tour_len"].item()


def main(argv: List[str] | None = None):
    from src.utils.lkh_solver import default_lkh_executable

    parser = argparse.ArgumentParser()
    default_lkh = default_lkh_executable()
    parser.add_argument("--ckpt", type=str, required=False, help="path to model checkpoint")
    parser.add_argument("--data_pt", type=str, required=False, help="path to data .pt (list of Data)")
    parser.add_argument("--sample_idx", type=int, default=0, help="start sample index")
    parser.add_argument("--sample_idx_end", type=int, default=None, help="end sample index (exclusive), if None, test only one")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--use_iface_in_decode", type=parse_bool_arg, default=True)
    parser.add_argument("--lkh_exe", type=str, default=default_lkh, help="path to LKH executable")
    parser.add_argument("--use_lkh", action="store_true", help="deprecated; teacher generation always uses LKH on the sparse spanner")
    parser.add_argument("--no_vis", action="store_true", help="disable visualization")
    parser.add_argument("--two_opt_passes", type=int, default=30, help="deprecated; ignored by spanner-LKH teacher generation")
    parser.add_argument("--teacher_lkh_runs", type=int, default=1)
    parser.add_argument("--teacher_lkh_timeout", type=float, default=0.0, help="0 disables timeout")
    parser.add_argument("--pomo_ckpt", type=str, default=None, help="path to POMO checkpoint")
    parser.add_argument("--neurolkh_ckpt", type=str, default=None, help="path to NeuroLKH checkpoint")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for parallel decoding")
    parser.add_argument("--run_exact", action="store_true", help="legacy flag: add exact to selected settings")
    parser.add_argument("--exact_time_limit", type=float, default=30.0, help="time limit in seconds for exact sparse decoding")
    parser.add_argument("--exact_length_weight", type=float, default=0.0, help="optional Euclidean tie-break weight for exact sparse decoding")
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="comma-separated settings/groups to evaluate; "
             + describe_settings(available=AVAILABLE_SETTINGS, groups=SETTING_GROUPS),
    )
    parser.add_argument("--list_settings", action="store_true", help="print available settings/groups and exit")
    args = parser.parse_args(argv)

    if args.list_settings:
        print(describe_settings(available=AVAILABLE_SETTINGS, groups=SETTING_GROUPS))
        return
    if not args.ckpt:
        parser.error("--ckpt is required unless --list_settings is used")
    if not args.data_pt:
        parser.error("--data_pt is required unless --list_settings is used")

    selected_settings = resolve_eval_settings(
        requested=args.settings,
        available=AVAILABLE_SETTINGS,
        default=DEFAULT_SETTINGS,
        aliases=SETTING_ALIASES,
        groups=SETTING_GROUPS,
        enable_exact=bool(args.run_exact),
    )
    print(f"[eval] selected settings: {', '.join(selected_settings)}")

    device = resolve_device(str(args.device))
    print(f"[env] device={device}")

    from src.models.bc_state_catalog import infer_boundary_state_count
    from src.models.bottom_up_runner import BottomUpTreeRunner
    from src.models.decode_backend import DecodingDataset
    from src.models.edge_aggregation import aggregate_logits_to_edges
    from src.models.labeler import PseudoLabeler
    from src.models.leaf_encoder import LeafEncoder
    from src.models.lkh_decode import solve_with_lkh_parallel
    from src.models.merge_encoder import MergeEncoder
    from src.models.node_token_packer import NodeTokenPacker
    from src.models.top_down_decoder import TopDownDecoder
    from src.models.top_down_runner import TopDownTreeRunner

    dataset = load_dataset(args.data_pt)
    start_idx = args.sample_idx
    end_idx = args.sample_idx_end if args.sample_idx_end is not None else start_idx + 1
    end_idx = min(end_idx, len(dataset))
    num_samples = end_idx - start_idx

    print(f"[ckpt] loading from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if "leaf_encoder" in ckpt and "emb_type.weight" in ckpt["leaf_encoder"]:
        d_model = ckpt["leaf_encoder"]["emb_type.weight"].shape[1]
    else:
        d_model = 128
    print(f"[model] detected d_model={d_model}")

    ckpt_args = ckpt.get("args", {})
    state_mode = str(ckpt_args.get("state_mode", "iface"))
    matching_max_used = int(ckpt_args.get("matching_max_used", 4))
    num_states = None
    if state_mode == "matching":
        num_states = infer_boundary_state_count(num_slots=4 * int(args.r), max_used=matching_max_used)

    leaf_encoder = LeafEncoder(d_model=d_model).to(device)
    merge_encoder = MergeEncoder(d_model=d_model).to(device)
    decoder = TopDownDecoder(
        d_model=d_model,
        mode=ckpt_args.get("td_mode", "two_stage"),
        state_mode=state_mode,
        num_states=num_states,
    ).to(device)
    leaf_encoder.load_state_dict(ckpt["leaf_encoder"])
    merge_encoder.load_state_dict(ckpt["merge_encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    leaf_encoder.eval()
    merge_encoder.eval()
    decoder.eval()
    print("[ckpt] loaded.\n")

    pomo_model = None
    pomo_env = None
    if "pomo" in selected_settings:
        if args.pomo_ckpt and os.path.exists(args.pomo_ckpt):
            print(f"[baseline] loading POMO from {args.pomo_ckpt}")
            pomo_model, pomo_env = load_pomo(device, args.pomo_ckpt)
            print("[baseline] POMO loaded.")
        else:
            print("[warn] POMO selected but --pomo_ckpt is missing or not found; POMO results will be N/A.")

    nlkh_model = None
    if "neurolkh" in selected_settings:
        if args.neurolkh_ckpt and os.path.exists(args.neurolkh_ckpt):
            print(f"[baseline] loading NeuroLKH from {args.neurolkh_ckpt}")
            nlkh_model = load_neurolkh(device, args.neurolkh_ckpt)
            print("[baseline] NeuroLKH loaded.")
        else:
            print("[warn] NeuroLKH selected but --neurolkh_ckpt is missing or not found; NeuroLKH results will be N/A.")

    stats = {setting: {"len": 0.0, "gap": 0.0, "time": 0.0, "cnt": 0} for setting in selected_settings}

    packer = NodeTokenPacker(
        r=int(args.r),
        state_mode=state_mode,
        matching_max_used=matching_max_used,
    )
    bu_runner = BottomUpTreeRunner()
    td_runner = TopDownTreeRunner()
    labeler = PseudoLabeler(
        two_opt_passes=int(args.two_opt_passes),
        use_lkh=bool(args.use_lkh),
        lkh_exe=str(args.lkh_exe),
        prefer_cpu=True,
        teacher_mode="spanner_lkh",
        teacher_lkh_runs=int(args.teacher_lkh_runs),
        teacher_lkh_timeout=(None if float(args.teacher_lkh_timeout) <= 0 else float(args.teacher_lkh_timeout)),
    )
    print(f"[teacher] using LKH executable: {labeler.lkh_exe}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[eval] Phase 1: Running neural models on {num_samples} samples...")
    model_outputs = []
    for offset, s_idx in enumerate(range(start_idx, end_idx), start=1):
        prefix = f"[eval {offset}/{num_samples}] sample={s_idx}"
        data = dataset[s_idx]
        data = move_data_tensors_to_device(data, device)
        log_progress(prefix, "run neural inference...")
        t0 = time.time()
        with torch.no_grad():
            packed = packer.pack_batch([data])
            t1 = time.time()
            out_bu = bu_runner.run_batch(batch=packed, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
            t2 = time.time()
            out_td = td_runner.run_batch(packed=packed, z=out_bu.z, decoder=decoder)
            t3 = time.time()
            edge_scores = aggregate_logits_to_edges(
                tokens=packed.tokens,
                cross_logit=out_td.cross_logit,
                iface_logit=out_td.iface_logit if args.use_iface_in_decode else None,
                reduce="mean",
                num_edges=data.spanner_edge_index.shape[1],
            )
            el = edge_scores.edge_logit.clone()
            em = edge_scores.edge_mask.bool()
            el[~em] = -1e9
            model_outputs.append(
                {
                    "s_idx": s_idx,
                    "pos": data.pos.cpu(),
                    "edge_index": data.spanner_edge_index.cpu(),
                    "edge_logit": el.cpu(),
                    "packed_tokens": packed.tokens,
                    "y_tour": data.y_tour.detach().cpu() if hasattr(data, "y_tour") and data.y_tour is not None else None,
                }
            )
        log_progress(
            prefix,
            "inference done: "
            f"pack={t1 - t0:.2f}s, bottom_up={t2 - t1:.2f}s, top_down={t3 - t2:.2f}s, aggregate={time.time() - t3:.2f}s",
        )

    print("[eval] Phase 2: Preparing parallel teacher calculations...")
    teacher_tasks = []
    for mo in model_outputs:
        ts = SimpleNamespace(
            cross_eid=mo["packed_tokens"].cross_eid.cpu(),
            cross_mask=mo["packed_tokens"].cross_mask.cpu(),
            iface_eid=mo["packed_tokens"].iface_eid.cpu(),
            iface_mask=mo["packed_tokens"].iface_mask.cpu(),
        )
        dummy_data = SimpleNamespace(
            pos=mo["pos"],
            spanner_edge_index=mo["edge_index"],
            spanner_edge_attr=torch.zeros(mo["edge_index"].shape[1], 1),
        )
        teacher_tasks.append((dummy_data, ts))

    t_dataset = TeacherDataset(teacher_tasks, labeler)
    teacher_workers = min(args.num_workers, len(teacher_tasks)) if teacher_tasks else 0
    t_loader = torch.utils.data.DataLoader(
        t_dataset,
        batch_size=1,
        num_workers=teacher_workers,
        shuffle=False,
        collate_fn=lambda x: x[0],
    )
    for i, t_len in enumerate(t_loader):
        model_outputs[i]["teacher_len"] = float(t_len.item()) if isinstance(t_len, torch.Tensor) else float(t_len)

    need_greedy = ("greedy" in selected_settings) or ("guided_lkh" in selected_settings)
    greedy_results = [placeholder_result() for _ in range(len(model_outputs))]
    exact_results = [placeholder_result() for _ in range(len(model_outputs))]

    if need_greedy:
        print("[eval] Phase 3: Running Greedy decoding in parallel...")
        greedy_tasks = [
            (mo["pos"], mo["edge_index"], mo["edge_logit"], True, True, mo["teacher_len"])
            for mo in model_outputs
        ]
        g_dataset = DecodingDataset(greedy_tasks)
        g_loader = torch.utils.data.DataLoader(
            g_dataset,
            batch_size=1,
            num_workers=min(args.num_workers, len(greedy_tasks)),
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
        for i, res_pair in enumerate(g_loader):
            greedy_results[i] = res_pair[0]

    if "exact" in selected_settings:
        print("[eval] Phase 4: Running Exact sparse decoding in parallel...")
        exact_tasks = [
            (mo["pos"], mo["edge_index"], mo["edge_logit"], True, False, mo["teacher_len"])
            for mo in model_outputs
        ]
        ex_dataset = DecodingDataset(
            exact_tasks,
            decode_backend="exact",
            exact_time_limit=float(args.exact_time_limit),
            exact_length_weight=float(args.exact_length_weight),
        )
        ex_loader = torch.utils.data.DataLoader(
            ex_dataset,
            batch_size=1,
            num_workers=min(args.num_workers, len(exact_tasks)),
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
        for i, res_pair in enumerate(ex_loader):
            exact_results[i] = res_pair[0]

    print("[eval] Phase 5: Running LKH and neural baselines...")
    for i, mo in enumerate(model_outputs):
        s_idx = mo["s_idx"]
        prefix = f"[eval {i + 1}/{num_samples}] sample={s_idx}"
        data_pos = mo["pos"]
        sp_edge_index = mo["edge_index"]
        el = mo["edge_logit"]
        y_tour = mo.get("y_tour")
        teacher_len = mo["teacher_len"]

        sample_results = {setting: placeholder_result() for setting in selected_settings}
        if "greedy" in selected_settings:
            sample_results["greedy"] = greedy_results[i]
        elif need_greedy:
            sample_results["greedy"] = greedy_results[i]

        if "exact" in selected_settings:
            sample_results["exact"] = exact_results[i]

        log_progress(prefix, "start post-processing / baselines...")

        lkh_tasks = []
        lkh_task_names: List[str] = []

        if "guided_lkh" in selected_settings:
            greedy_res = greedy_results[i]
            lkh_tasks.append(
                {
                    "pos": data_pos,
                    "mode": "guided",
                    "edge_index": sp_edge_index,
                    "edge_logit": el,
                    "teacher_len": teacher_len,
                    "initial_tour": greedy_res.order if greedy_res.feasible else None,
                }
            )
            lkh_task_names.append("guided_lkh")

        if "pure_lkh" in selected_settings:
            lkh_tasks.append({"pos": data_pos, "mode": "pure", "teacher_len": teacher_len})
            lkh_task_names.append("pure_lkh")

        if "neurolkh" in selected_settings and nlkh_model is not None:
            with torch.no_grad():
                n_nodes = data_pos.shape[0]
                n_edges = min(20, max(int(n_nodes) - 1, 1))
                node_feat = data_pos.unsqueeze(0).to(device) / 10000.0
                dist = torch.cdist(node_feat, node_feat)
                dist.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
                topk_values, topk_indices = torch.topk(dist, n_edges, dim=2, largest=False)
                edge_index = topk_indices.view(1, -1).to(device)
                edge_feat = topk_values.view(1, -1, 1).to(device)

                target_nodes_neighbors = torch.gather(
                    topk_indices,
                    1,
                    topk_indices.view(1, n_nodes * n_edges).unsqueeze(-1).expand(-1, -1, n_edges),
                ).view(1, n_nodes, n_edges, n_edges)
                current_nodes_expanded = torch.arange(n_nodes, device=device).view(1, n_nodes, 1, 1).expand(1, -1, n_edges, n_edges)
                matches_mask = target_nodes_neighbors == current_nodes_expanded
                k_indices = matches_mask.max(dim=3)[1]
                inverse_edge_index = (topk_indices * n_edges + k_indices).view(1, -1)
                no_match_mask = ~matches_mask.max(dim=3)[0]
                inverse_edge_index.view(1, n_nodes, n_edges)[no_match_mask] = n_nodes * n_edges

                y_pred_edges_log, _, _ = nlkh_model(node_feat, edge_feat, edge_index, inverse_edge_index, None, None, n_edges)
                nlkh_edge_logit = y_pred_edges_log[0, :, 1].view(n_nodes, n_edges)
                nlkh_spanner_index = torch.stack(
                    [
                        torch.arange(n_nodes, device=device).unsqueeze(1).expand(-1, n_edges).reshape(-1),
                        topk_indices.view(-1),
                    ],
                    dim=0,
                )

                lkh_tasks.append(
                    {
                        "pos": data_pos,
                        "mode": "guided",
                        "edge_index": nlkh_spanner_index.cpu(),
                        "edge_logit": nlkh_edge_logit.view(-1).cpu(),
                        "teacher_len": teacher_len,
                    }
                )
                lkh_task_names.append("neurolkh")

        if lkh_tasks:
            log_progress(prefix, f"run LKH-based settings: {', '.join(lkh_task_names)}")
            lkh_results_all = solve_with_lkh_parallel(
                lkh_tasks,
                lkh_executable=getattr(args, "lkh_exe", "LKH"),
                num_workers=max(1, min(args.num_workers, len(lkh_tasks))),
            )
            for setting_name, (res, _) in zip(lkh_task_names, lkh_results_all):
                sample_results[setting_name] = res

        if "pomo" in selected_settings and pomo_model is not None and pomo_env is not None:
            log_progress(prefix, "run POMO...")
            t_pomo = time.time()
            with torch.no_grad():
                n_nodes = data_pos.shape[0]
                pomo_env.problem_size = n_nodes
                pomo_env.pomo_size = n_nodes
                pomo_env.set_problems(data_pos.unsqueeze(0).to(device) / 10000.0)
                reset_state, _, _ = pomo_env.reset()
                pomo_model.pre_forward(reset_state)
                state, reward, done = pomo_env.pre_step()
                while not done:
                    selected, _ = pomo_model(state)
                    state, reward, done = pomo_env.step(selected)

                best_reward, best_idx = reward.max(dim=1)
                pomo_res = placeholder_result()
                pomo_res.length = -best_reward.item() * 10000.0
                pomo_res.feasible = True
                if device.type == "cuda":
                    torch.cuda.synchronize()
                pomo_res.duration = time.time() - t_pomo
                pomo_res.order = pomo_env.selected_node_list[0, best_idx.item(), :]
                sample_results["pomo"] = pomo_res
            log_progress(prefix, f"POMO done in {pomo_res.duration:.2f}s")

        def update_stats(key, res):
            if res.feasible and res.length < float("inf"):
                gap = (res.length / teacher_len - 1.0) if teacher_len > 1e-9 else 0.0
                stats[key]["len"] += res.length
                stats[key]["gap"] += gap
                stats[key]["time"] += getattr(res, "duration", 0.0)
                stats[key]["cnt"] += 1
                return gap
            return float("inf")

        sample_gaps = {}
        for setting in selected_settings:
            result = sample_results[setting]
            sample_gaps[setting] = update_stats(setting, result)

        for setting in selected_settings:
            result = sample_results[setting]
            gap = sample_gaps[setting]
            gap_text = "N/A" if not math.isfinite(gap) else f"{gap * 100: >7.4f}%"
            print(f"  {SETTING_DISPLAY_NAMES[setting]:<15} Gap: {gap_text:>9} Time: {getattr(result, 'duration', 0.0):.3f}s")
        log_progress(prefix, "sample complete")

        if not args.no_vis:
            for setting in selected_settings:
                result = sample_results[setting]
                gap = sample_gaps[setting]
                if not result.feasible:
                    continue

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.set_aspect("equal")
                pos_np = data_pos.detach().cpu().numpy()
                ax.scatter(pos_np[:, 0], pos_np[:, 1], s=20, c="black", zorder=5)

                if y_tour is not None:
                    plot_tour(
                        data_pos,
                        prepare_edges(y_tour, data_pos.device),
                        ax,
                        color="gray",
                        lw=2,
                        alpha=0.1,
                        label="Teacher Reference",
                        linestyle="--",
                    )

                plot_tour(
                    data_pos,
                    prepare_edges(result.order, data_pos.device),
                    ax,
                    color=SETTING_COLORS[setting],
                    lw=1.5,
                    label=f"{SETTING_DISPLAY_NAMES[setting]} (Gap: {gap * 100:.2f}%)",
                )
                ax.legend()
                ax.set_title(f"TSP {SETTING_DISPLAY_NAMES[setting]} - Sample {s_idx}\nGap: {gap * 100:.4f}%")
                plt.savefig(output_dir / f"vis_sample_{s_idx}_{setting}.png", dpi=150)
                plt.close()

    print(f"\n{'=' * 20} FINAL SUMMARY ({num_samples} samples) {'=' * 20}")
    print(f"{'Method':<15} | {'Avg Length':<12} | {'Avg Gap (%)':<12} | {'Total Time':<10}")
    print("-" * 65)
    for setting in selected_settings:
        m = stats[setting]
        if m["cnt"] > 0:
            avg_len = m["len"] / m["cnt"]
            avg_gap = (m["gap"] / m["cnt"]) * 100
            print(f"{SETTING_DISPLAY_NAMES[setting]:<15} | {avg_len:<12.4f} | {avg_gap:<12.4f} | {m['time']:<10.3f}s")
        else:
            print(f"{SETTING_DISPLAY_NAMES[setting]:<15} | {'N/A':<12} | {'N/A':<12} | {'N/A':<10}")
    print("=" * 65)


if __name__ == "__main__":
    main()
