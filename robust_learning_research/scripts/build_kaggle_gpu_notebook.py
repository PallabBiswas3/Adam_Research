"""Generate the self-contained Kaggle notebook from reviewed local sources."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle" / "robust_learning_gpu_pilot.ipynb"
EMBEDDED = (
    "configs/strong_baseline_pilot.json",
    "src/robust_learning/__init__.py",
    "src/robust_learning/strong/__init__.py",
    "src/robust_learning/strong/cifar.py",
    "src/robust_learning/strong/equations.py",
    "src/robust_learning/strong/kernels.py",
    "src/robust_learning/strong/vendor_dividemix/PreResNet.py",
    "src/robust_learning/strong/vendor_dividemix/LICENSE",
    "scripts/run_kaggle_gpu_pilot.py",
)


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def make_notebook():
    cells = [markdown("""# Robust Learning: bounded CIFAR-10 GPU timing pilot

Runs one full-data warmup epoch and one mature-operation epoch for DSS+ and DivideMix, plus DivideMix's full-data GMM scan. The entire pilot has a shared 30-minute deadline and records censored stages. It measures timing and GPU memory only; it does not evaluate accuracy, selection quality, or research hypotheses.

Before running, attach an existing Kaggle dataset containing the verified official `cifar-10-batches-py` directory. Enable GPU. No internet, credentials, Kaggle CLI, dataset upload, or dependency installation is used by this notebook. Review `research/kaggle_gpu_pilot.md` in the repository for fidelity decisions and limitations. The bundled backbone is the MIT-licensed DivideMix `PreResNet.py`; the original license is embedded alongside it. DSS source is not bundled.
""")]
    manifest = []
    for relative in EMBEDDED:
        body = (ROOT / relative).read_text(encoding="utf-8")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        manifest.append({"path": relative, "sha256": digest})
        cells.append(code("from pathlib import Path\n"
                          "destination = Path('/kaggle/working/robust_learning_research') / " + repr(relative) + "\n"
                          "destination.parent.mkdir(parents=True, exist_ok=True)\n"
                          "destination.write_text(" + repr(body) + ", encoding='utf-8')\n"))
    cells.append(code("""import hashlib
import json
from pathlib import Path

project = Path('/kaggle/working/robust_learning_research')
manifest = """ + repr(manifest) + """
for item in manifest:
    actual = hashlib.sha256((project / item['path']).read_bytes()).hexdigest()
    assert actual == item['sha256'], item['path']
print('Embedded source hashes verified:', len(manifest))
"""))
    cells.append(code("""from pathlib import Path
import sys

# Edit this if more than one attached dataset contains the official CIFAR-10 batches.
candidates = sorted(p.parent for p in Path('/kaggle/input').rglob('cifar-10-batches-py') if p.is_dir())
if len(candidates) != 1:
    raise RuntimeError(f'Expected exactly one attached extracted CIFAR-10 dataset; found {candidates}')
cifar_root = candidates[0]
sys.path.insert(0, '/kaggle/working/robust_learning_research/scripts')
from run_kaggle_gpu_pilot import run
from argparse import Namespace

report = run(Namespace(cifar_root=cifar_root,
                       output=Path('/kaggle/working/robust-learning-gpu-pilot'),
                       batch_size=128, time_limit_seconds=1800))
print('Report:', '/kaggle/working/robust-learning-gpu-pilot/gpu_pilot_report.json')
print({name: entry['status'] for name, entry in report['methods'].items()})
"""))
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(make_notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(DEST)
