import subprocess
import sys

from ultralytics.utils import MACOS
from ultralytics.utils.dist import ddp_launch_env, find_free_network_port
from ultralytics.utils.torch_utils import TORCH_1_9


MARKER = "DDP_E2E_UNIQUE_MARKER_7F3A"


def test_torchrun_rank1_marker_reaches_parent_output(tmp_path):
    worker = tmp_path / "worker.py"
    logs = tmp_path / "logs"
    elastic_import = "from torch.distributed.elastic.multiprocessing.errors import record\n@record\n" if TORCH_1_9 else ""
    worker_source = "import os\n" + elastic_import + (
        "def main():\n"
        "    if int(os.environ['LOCAL_RANK']) == 1:\n"
        f"        raise RuntimeError('{MARKER}')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    worker.write_text(worker_source, encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch",
        "--master_addr=127.0.0.1",
        f"--master_port={find_free_network_port()}",
        "--nproc_per_node=2",
    ]
    if TORCH_1_9:
        command.extend(["--log-dir", str(logs)])
        if not MACOS:
            command.extend(["--tee", "3"])
    command.append(str(worker))
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90, env=ddp_launch_env()
    )
    assert result.returncode != 0
    assert MARKER in result.stdout
    if TORCH_1_9:
        assert "Root Cause" in result.stdout
        assert list(logs.rglob("error.json"))
