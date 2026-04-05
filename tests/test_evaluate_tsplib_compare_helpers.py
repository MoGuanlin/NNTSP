from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.evaluate_tsplib import select_tsplib_files
from src.experiments.evaluate_tsplib_compare import load_tsplib_optima, tsplib_euc2d_length


def test_n100_500_preset_selects_expected_range():
    all_tsp_files = [
        (52, Path("berlin52.tsp")),
        (100, Path("kroA100.tsp")),
        (150, Path("ch150.tsp")),
        (318, Path("lin318.tsp")),
        (500, Path("demo500.tsp")),
        (561, Path("pa561.tsp")),
    ]

    selected, desc, tag = select_tsplib_files(
        all_tsp_files=all_tsp_files,
        instance_preset="n100_500",
        instance_names=None,
        num_instances=10,
    )

    assert [n for n, _ in selected] == [100, 150, 318, 500]
    assert "100 <= N <= 500" in desc
    assert tag == "n100_500"


def test_load_tsplib_optima_and_euc2d_length(tmp_path: Path):
    tsplib_dir = tmp_path / "tsplib"
    tsplib_dir.mkdir()
    (tsplib_dir / "solutions").write_text(
        "berlin52 : 7542\n"
        "eil101 : 629\n",
        encoding="utf-8",
    )

    optima = load_tsplib_optima(tsplib_dir)
    assert optima["berlin52"] == 7542.0
    assert optima["eil101"] == 629.0

    pos = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert tsplib_euc2d_length(pos, [0, 1, 2, 3]) == 4.0
