"""Static checks for the portable notebook and its bounded runner; no GPU run."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_full_epoch_batches_cover_every_image_once():
    script = ROOT / "scripts" / "run_kaggle_gpu_pilot.py"
    spec = importlib.util.spec_from_file_location("run_kaggle_gpu_pilot", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    order = np.arange(18388)
    parts = list(module.batches(order, 128))
    assert len(parts) == 144
    assert len(parts[-1]) == 84
    np.testing.assert_array_equal(np.concatenate(parts), order)


def test_notebook_syntax_hashes_and_private_metadata():
    from scripts.build_kaggle_gpu_notebook import EMBEDDED, make_notebook

    notebook = json.loads((ROOT / "kaggle" / "robust_learning_gpu_pilot.ipynb").read_text(encoding="utf-8"))
    assert notebook == make_notebook()
    assert notebook["nbformat"] == 4
    assert all(not cell["outputs"] and cell["execution_count"] is None
               for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    launch = "".join(notebook["cells"][-1]["source"])
    assert "subprocess.run(command, timeout=1800" in launch
    assert "'--time-limit-seconds', '1770'" in launch
    assert len(EMBEDDED) == 9
    metadata = json.loads((ROOT / "kaggle" / "kernel-metadata.json").read_text())
    assert metadata["id"] == "pallabbiswas3/robust-learning-gpu-pilot"
    assert metadata["code_file"] == "robust_learning_gpu_pilot.ipynb"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == []
