"""Launch authoritative training workers with argument arrays and no shell."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


class TrainingProcessLauncher:
    def launch(self, job_directory: str | Path) -> int:
        directory = Path(job_directory).resolve()
        pid_path = directory / "worker.pid"
        if pid_path.exists():
            pid = int(pid_path.read_text(encoding="ascii"))
            if self.process_exists(pid):
                return pid
        log = (directory / "console.log").open("a", encoding="utf-8")
        kwargs = {"cwd": str(Path(__file__).resolve().parents[2]), "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "shell": False}
        if os.name == "nt": kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        process = subprocess.Popen(
            [sys.executable, "-m", "Rock_AI.scripts.run_neat_training_job", "--job", str(directory)],
            **kwargs,
        )
        log.close()
        pid_path.write_text(str(process.pid), encoding="ascii")
        return int(process.pid)

    @staticmethod
    def process_exists(pid: int) -> bool:
        if pid <= 0: return False
        if os.name == "nt":
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, shell=False)
            return str(pid) in result.stdout
        try: os.kill(pid, 0); return True
        except OSError: return False
