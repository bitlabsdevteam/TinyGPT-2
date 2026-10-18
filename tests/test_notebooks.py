"""Seam: the Colab pretraining notebook, executed top to bottom on a tiny config (ticket 10).

The notebook is executed headlessly with its SMOKE_TEST switch on (the 20M `small` preset,
a ~2 minute dry run on the validation split) and with a scratch work folder in place of
Google Drive, exactly as a human would do a dry run before the long A100 run.
"""
import json
import os
import socket
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("ipykernel")
from nbclient import NotebookClient  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "pretrain.ipynb"
RUN_NAME = "pretrain-gpt2-small-smoke"


def network_available() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


def execute(workdir: Path) -> None:
    os.environ["TINYGPT_SMOKE_TEST"] = "True"
    os.environ["TINYGPT_WORKDIR"] = str(workdir)
    try:
        nb = nbformat.read(NOTEBOOK, as_version=4)
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(REPO)}}).execute()
    finally:
        os.environ.pop("TINYGPT_SMOKE_TEST", None)
        os.environ.pop("TINYGPT_WORKDIR", None)


@pytest.fixture(scope="module")
def smoke_workdir(tmp_path_factory):
    if not network_available():
        pytest.skip("the notebook downloads TinyStories; no network")
    workdir = tmp_path_factory.mktemp("drive")
    execute(workdir)
    return workdir


def test_notebook_is_committed_without_outputs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert code_cells
    assert all(not c.outputs and c.execution_count is None for c in code_cells)


def test_notebook_runs_top_to_bottom_and_leaves_the_report_and_checkpoints(smoke_workdir):
    run = smoke_workdir / "runs" / RUN_NAME
    report = json.loads((run / "profile_report.json").read_text())
    assert report["summary"]["num_steps"] > 0
    assert (run / "loss_curves.png").exists()
    assert (run / "config.json").exists()
    assert list((run / "checkpoints").glob("ckpt_*.pt"))
    # The dry run's data lives apart from the real data, so it can never be mistaken for it.
    assert (smoke_workdir / "data_smoke").is_dir()
    assert not (smoke_workdir / "data").exists()


def test_running_the_notebook_again_resumes_instead_of_starting_over(smoke_workdir):
    run = smoke_workdir / "runs" / RUN_NAME
    steps = json.loads((run / "profile_report.json").read_text())["summary"]["num_steps"]

    execute(smoke_workdir)  # "Run all" again, as after a dropped session

    config = json.loads((run / "config.json").read_text())
    assert config["resumed_from_step"] == steps == config["num_steps"]
    assert json.loads((run / "profile_report.json").read_text())["summary"]["num_steps"] == steps
