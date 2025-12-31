# train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    # Avoid touching PyG deprecated virtual attrs (e.g., num_faces)
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


def load_list_pt(path: str) -> List[Any]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, list):
        raise RuntimeError(f"{path} is not a list[Data].")
    return obj


@torch.no_grad()
def run_validation(
    *,
    val_list: List[Any],
    device: torch.device,
    packer,
    leaf_encoder,
    merge_encoder,
    decoder,
    bu_runner,
    td_runner,
    labeler,
    batch_size: int,
    # loss weights
    w_token: float,
    w_iface_aux: float,
    w_bc: float,
) -> Dict[str, float]:
    """
    Validation returns:
      - token-level BCE losses (cross + optional iface aux)
      - BC loss (supervise out_td.bc_iface_logit vs teacher y_iface on non-root nodes)
      - teacher projection stats (direct/projected/unreachable)
      - decode metrics from edge_logit:
          feasible_rate, gap_mean (vs teacher), off_spanner_mean, components0_mean
    """
    from src.models.edge_aggregation import aggregate_cross_logits_to_edges
    from src.models.edge_decode import decode_tour_from_edge_logits
    from src.models.losses import dp_token_losses, masked_bce_with_logits

    leaf_encoder.eval()
    merge_encoder.eval()
    decoder.eval()

    # losses
    loss_total_list: List[float] = []
    loss_token_list: List[float] = []
    loss_cross_list: List[float] = []
    loss_iface_list: List[float] = []
    loss_bc_list: List[float] = []

    # teacher projection stats (per graph)
    direct_list: List[int] = []
    proj_list: List[int] = []
    unr_list: List[int] = []

    # decode metrics
    dec_feasible: List[float] = []
    dec_gap: List[float] = []
    dec_off: List[float] = []
    dec_comp0: List[float] = []

    for i in range(0, len(val_list), batch_size):
        datas = val_list[i : i + batch_size]
        datas = [move_data_tensors_to_device(d, device) for d in datas]

        packed = packer.pack_batch(datas)

        out_bu = bu_runner.run_batch(batch=packed, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
        z = out_bu.z
        out_td = td_runner.run_batch(packed=packed, z=z, decoder=decoder)

        # teacher labels (batch-level; includes y_cross/y_iface and child-bc targets)
        labels = labeler.label_batch(datas=datas, packed=packed, device=device)

        # token loss (cross + optional iface)
        token_out = dp_token_losses(
            cross_logit=out_td.cross_logit,
            y_cross=labels.y_cross,
            m_cross=labels.m_cross,
            iface_logit=out_td.iface_logit,
            y_iface=labels.y_iface,
            m_iface=labels.m_iface,
            w_iface=float(w_iface_aux),
            pos_weight_cross=None,
        )
        L_token = token_out.loss

        # BC loss: supervise bc_iface_logit on NON-root nodes against teacher y_iface
        total_M = int(out_td.bc_iface_logit.shape[0])
        root_ids = packed.tokens.root_id.long().to(device)
        is_root = torch.zeros((total_M,), device=device, dtype=torch.bool)
        is_root[root_ids] = True
        bc_mask = labels.m_iface & (~is_root.unsqueeze(1))  # [M,Ti]
        L_bc = masked_bce_with_logits(out_td.bc_iface_logit, labels.y_iface, bc_mask, pos_weight=None)

        loss = float(w_token) * L_token + float(w_bc) * L_bc

        loss_total_list.append(float(loss.item()))
        loss_token_list.append(float(L_token.item()))
        loss_cross_list.append(float(token_out.parts["loss_cross"].item()))
        loss_iface_list.append(float(token_out.parts.get("loss_iface", torch.tensor(0.0, device=device)).item()))
        loss_bc_list.append(float(L_bc.item()))

        # stats per graph (direct/proj/unr + teacher len for gap)
        teacher_len_per_graph: List[float] = []
        B = int(packed.node_ptr.numel() - 1)
        for b in range(B):
            # compute per-graph teacher stats via label_one (cheap enough; keeps interface stable)
            lo = int(packed.node_ptr[b].item())
            hi = int(packed.node_ptr[b + 1].item())
            eid_off = int(packed.edge_ptr[b].item())
            t = packed.tokens

            class _Slice:
                pass

            ts = _Slice()
            for name in ["cross_eid", "cross_mask", "iface_eid", "iface_mask"]:
                setattr(ts, name, getattr(t, name)[lo:hi])

            lab = labeler.label_one(data=datas[b], tokens_slice=ts, device=device, eid_offset=eid_off)
            direct_list.append(int(lab.stats["num_direct"].item()))
            proj_list.append(int(lab.stats["num_projected"].item()))
            unr_list.append(int(lab.stats["num_unreachable"].item()))
            teacher_len_per_graph.append(float(lab.stats["tour_len"].item()))

        # ---------- decode metrics ----------
        edge_scores = aggregate_cross_logits_to_edges(tokens=packed.tokens, cross_logit=out_td.cross_logit)
        edge_logit_g = edge_scores.edge_logit
        edge_mask_g = edge_scores.edge_mask.bool()

        for b in range(B):
            e0 = int(packed.edge_ptr[b].item())
            e1 = int(packed.edge_ptr[b + 1].item())
            if e1 <= e0:
                continue

            # local edge logits for this graph
            el = edge_logit_g[e0:e1].clone()
            em = edge_mask_g[e0:e1]

            # uncovered edges: push down so decode prefers covered edges
            el[~em] = -1e9

            res = decode_tour_from_edge_logits(
                pos=datas[b].pos,
                spanner_edge_index=datas[b].spanner_edge_index,
                edge_logit=el,
                prefer_spanner_only=True,
                allow_off_spanner_patch=True,
            )

            dec_feasible.append(1.0 if res.feasible else 0.0)
            dec_off.append(float(res.num_off_spanner_edges))
            dec_comp0.append(float(res.num_components_initial))

            tlen = teacher_len_per_graph[b]
            gap = (res.length / tlen - 1.0) if tlen > 1e-9 else 0.0
            dec_gap.append(float(gap))

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)

    return {
        "val_loss": avg(loss_total_list),
        "val_loss_token": avg(loss_token_list),
        "val_loss_cross": avg(loss_cross_list),
        "val_loss_iface": avg(loss_iface_list),
        "val_loss_bc": avg(loss_bc_list),
        "val_direct": avg([float(x) for x in direct_list]) if direct_list else 0.0,
        "val_projected": avg([float(x) for x in proj_list]) if proj_list else 0.0,
        "val_unreachable": avg([float(x) for x in unr_list]) if unr_list else 0.0,
        "val_decode_feasible_rate": avg(dec_feasible) if dec_feasible else 0.0,
        "val_decode_gap_mean": avg(dec_gap) if dec_gap else 0.0,
        "val_decode_off_spanner_mean": avg(dec_off) if dec_off else 0.0,
        "val_decode_components0_mean": avg(dec_comp0) if dec_comp0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, default="")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=200)

    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=500)

    # Teacher / labeler
    parser.add_argument("--two_opt_passes", type=int, default=30)

    # Top-down decoder
    parser.add_argument("--td_mode", type=str, default="two_stage", choices=["two_stage", "one_stage"])

    # Loss weights
    parser.add_argument("--w_token", type=float, default=1.0, help="weight for token-level losses (cross + iface aux)")
    parser.add_argument("--w_iface_aux", type=float, default=0.05, help="aux weight for iface token BCE inside token loss")
    parser.add_argument("--w_bc", type=float, default=1.0, help="weight for BC loss on out_td.bc_iface_logit")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[env] device={device}")
    set_seed(int(args.seed))

    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.bottom_up_runner import BottomUpTreeRunner
    from src.models.top_down_decoder import TopDownDecoder
    from src.models.top_down_runner import TopDownTreeRunner
    from src.models.labeler import PseudoLabeler
    from src.models.losses import dp_token_losses, masked_bce_with_logits

    train_list = load_list_pt(args.train_pt)
    val_list = load_list_pt(args.val_pt) if args.val_pt else []

    packer = NodeTokenPacker(r=int(args.r))

    leaf_encoder = LeafEncoder(d_model=128).to(device)
    merge_encoder = MergeEncoder(d_model=128).to(device)

    # NOTE: new TopDownDecoder no longer takes "r=..."; mode switch here.
    decoder = TopDownDecoder(d_model=128, mode=str(args.td_mode)).to(device)

    bu_runner = BottomUpTreeRunner()
    td_runner = TopDownTreeRunner()

    labeler = PseudoLabeler(two_opt_passes=int(args.two_opt_passes), prefer_cpu=True)

    params = list(leaf_encoder.parameters()) + list(merge_encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.wd))

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    t0 = time.time()

    for epoch in range(int(args.epochs)):
        leaf_encoder.train()
        merge_encoder.train()
        decoder.train()

        idx = list(range(len(train_list)))
        random.shuffle(idx)

        for it in range(0, len(idx), int(args.batch_size)):
            batch_ids = idx[it : it + int(args.batch_size)]
            datas = [train_list[j] for j in batch_ids]
            datas = [move_data_tensors_to_device(d, device) for d in datas]

            packed = packer.pack_batch(datas)

            out_bu = bu_runner.run_batch(batch=packed, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
            z = out_bu.z
            out_td = td_runner.run_batch(packed=packed, z=z, decoder=decoder)

            # batch teacher labels (includes y_cross/y_iface and y_child_iface)
            labels = labeler.label_batch(datas=datas, packed=packed, device=device)

            # token loss (cross + optional iface aux)
            token_out = dp_token_losses(
                cross_logit=out_td.cross_logit,
                y_cross=labels.y_cross,
                m_cross=labels.m_cross,
                iface_logit=out_td.iface_logit,
                y_iface=labels.y_iface,
                m_iface=labels.m_iface,
                w_iface=float(args.w_iface_aux),
                pos_weight_cross=None,
            )
            L_token = token_out.loss

            # BC loss: out_td.bc_iface_logit is exactly the parent's prediction for each non-root node
            total_M = int(out_td.bc_iface_logit.shape[0])
            root_ids = packed.tokens.root_id.long().to(device)
            is_root = torch.zeros((total_M,), device=device, dtype=torch.bool)
            is_root[root_ids] = True
            bc_mask = labels.m_iface & (~is_root.unsqueeze(1))
            L_bc = masked_bce_with_logits(out_td.bc_iface_logit, labels.y_iface, bc_mask, pos_weight=None)

            loss = float(args.w_token) * L_token + float(args.w_bc) * L_bc

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))

            opt.step()
            global_step += 1

            # optional: keep b0 teacher projection stats for logging only
            direct0 = proj0 = unr0 = None
            if packed.node_ptr.numel() > 1:
                lo0 = int(packed.node_ptr[0].item())
                hi0 = int(packed.node_ptr[1].item())
                eid_off0 = int(packed.edge_ptr[0].item())
                t = packed.tokens

                class _Slice:
                    pass

                ts = _Slice()
                for name in ["cross_eid", "cross_mask", "iface_eid", "iface_mask"]:
                    setattr(ts, name, getattr(t, name)[lo0:hi0])

                lab0 = labeler.label_one(data=datas[0], tokens_slice=ts, device=device, eid_offset=eid_off0)
                direct0 = int(lab0.stats["num_direct"].item())
                proj0 = int(lab0.stats["num_projected"].item())
                unr0 = int(lab0.stats["num_unreachable"].item())

            if global_step % int(args.log_interval) == 0:
                dt = time.time() - t0
                parts = ", ".join([f"{k}={float(v.item()):.6f}" for k, v in token_out.parts.items()])
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={float(loss.item()):.6f} "
                    f"L_token={float(L_token.item()):.6f} L_bc={float(L_bc.item()):.6f} "
                    f"({parts}) "
                    f"b0(direct={direct0},proj={proj0},unr={unr0}) "
                    f"time={dt:.1f}s"
                )

            if global_step % int(args.save_interval) == 0:
                ckpt_path = ckpt_dir / f"ckpt_step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                        "leaf_encoder": leaf_encoder.state_dict(),
                        "merge_encoder": merge_encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "opt": opt.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved to {ckpt_path}")

            if val_list and (global_step % int(args.val_interval) == 0):
                st = run_validation(
                    val_list=val_list,
                    device=device,
                    packer=packer,
                    leaf_encoder=leaf_encoder,
                    merge_encoder=merge_encoder,
                    decoder=decoder,
                    bu_runner=bu_runner,
                    td_runner=td_runner,
                    labeler=labeler,
                    batch_size=int(args.batch_size),
                    w_token=float(args.w_token),
                    w_iface_aux=float(args.w_iface_aux),
                    w_bc=float(args.w_bc),
                )
                msg = " ".join([f"{k}={v:.6f}" for k, v in st.items()])
                print(f"[val] {msg}")

    ckpt_path = ckpt_dir / f"ckpt_final_step_{global_step}.pt"
    torch.save(
        {
            "step": global_step,
            "epoch": int(args.epochs),
            "args": vars(args),
            "leaf_encoder": leaf_encoder.state_dict(),
            "merge_encoder": merge_encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "opt": opt.state_dict(),
        },
        ckpt_path,
    )
    print(f"[ckpt] saved to {ckpt_path}")
    print("[done] training finished.")


if __name__ == "__main__":
    main()
