import torch

from src.models.lkh_decode import build_guided_candidates, build_uniform_spanner_candidates


def test_build_guided_candidates_keeps_topk_highest_scores_per_node():
    edge_index = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 2, 3, 2],
        ],
        dtype=torch.long,
    )
    edge_logit = torch.tensor([0.9, 0.1, 0.5, 0.2], dtype=torch.float32)

    candidates = build_guided_candidates(
        num_nodes=4,
        edge_index=edge_index,
        edge_logit=edge_logit,
        logit_scale=1000.0,
        top_k=2,
    )

    assert candidates[0] == [(1, 0), (3, 400)]
    assert candidates[1] == [(0, 0), (2, 700)]
    assert candidates[2] == [(1, 700), (0, 800)]
    assert candidates[3] == [(0, 400)]


def test_build_uniform_spanner_candidates_uses_all_edges_with_same_alpha():
    edge_index = torch.tensor(
        [
            [0, 0, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    candidates = build_uniform_spanner_candidates(
        num_nodes=4,
        edge_index=edge_index,
        uniform_alpha=7,
    )

    assert candidates[0] == [(1, 7), (2, 7)]
    assert candidates[1] == [(0, 7)]
    assert candidates[2] == [(0, 7), (3, 7)]
    assert candidates[3] == [(2, 7)]
