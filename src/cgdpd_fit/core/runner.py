"""Run LAMMPS, several independent simulations at a time."""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi", "lmp_omp", "lammps")


def find_lammps() -> str | None:
    env = os.environ.get("LAMMPS_EXE")
    if env and shutil.which(env):
        return shutil.which(env)
    for c in CANDIDATES:
        p = shutil.which(c)
        if p:
            return p
    for p in (Path.home() / "lammps-install/bin/lmp", Path.home() / ".local/bin/lmp"):
        if p.exists():
            return str(p)
    return None


def kill_process_tree(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Terminate a process started in its own session, including children (mpirun ranks)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":                 # no process groups on Windows
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


@dataclass
class RunSettings:
    lammps_exe: str = ""             # empty: search PATH / ~/lammps-install
    prefix: str = ""                 # e.g. "mpirun -np 2"
    extra_args: str = ""             # e.g. "-sf omp -pk omp 2"
    n_jobs: int = 4                  # simultaneous LAMMPS runs
    threads: int = 1                 # OMP_NUM_THREADS of every run
    timeout: float = 0.0             # seconds, 0 = none
    keep_trajectories: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunSettings":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        return s

    def exe(self) -> str:
        exe = self.lammps_exe or find_lammps()
        if not exe:
            raise FileNotFoundError("LAMMPS executable not found; set it in the run settings "
                                    "or the LAMMPS_EXE environment variable")
        return exe

    def command(self, script: str, threads: int | None = None) -> list[str]:
        threads = self.threads if threads is None else threads
        extra = shlex.split(self.extra_args)
        if not extra and threads > 1:
            extra = ["-sf", "omp", "-pk", "omp", str(threads)]
        return shlex.split(self.prefix) + [self.exe(), "-in", script, "-log", "log.lammps",
                                           "-screen", "none"] + extra


class Cancelled(RuntimeError):
    """Raised when the user stops a running job."""


class LammpsPool:
    """Runs batches of LAMMPS jobs; :meth:`stop` kills everything that is running."""

    def __init__(self, settings: RunSettings):
        self.settings = settings
        self._procs: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self.stopped = False
        self._batch_threads = max(1, int(settings.threads))

    def stop(self) -> None:
        self.stopped = True
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            kill_process_tree(p)

    def run_one(self, workdir: Path, script: str = "in.cg") -> tuple[int, str]:
        """Run LAMMPS in ``workdir``; returns (exit code, last lines of the output)."""
        if self.stopped:
            raise Cancelled()
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(self._batch_threads)
        cmd = self.settings.command(script, self._batch_threads)
        out = Path(workdir) / "lammps.out"
        with open(out, "w") as fh:
            proc = subprocess.Popen(cmd, cwd=workdir, stdout=fh, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            with self._lock:
                self._procs.add(proc)
            try:
                rc = proc.wait(self.settings.timeout or None)
            except subprocess.TimeoutExpired:
                kill_process_tree(proc)
                rc = -9
            finally:
                with self._lock:
                    self._procs.discard(proc)
        if self.stopped:
            raise Cancelled()
        tail = ""
        for name in ("log.lammps", "lammps.out"):
            p = Path(workdir) / name
            if p.exists():
                lines = p.read_text(errors="replace").splitlines()
                err = [ln for ln in lines if "ERROR" in ln]
                tail = "\n".join(err[-3:] or lines[-8:])
                if err:
                    break
        return rc, tail

    def map(self, fn, items) -> list:
        """Apply ``fn`` (which calls :meth:`run_one`) to ``items`` with ``n_jobs`` workers."""
        items = list(items)
        n = max(1, min(int(self.settings.n_jobs), len(items) or 1))
        # cores left idle by a small batch (e.g. one IBI run) go to OpenMP threads,
        # unless the user gave explicit LAMMPS accelerator arguments
        total = max(1, int(self.settings.n_jobs)) * max(1, int(self.settings.threads))
        self._batch_threads = (max(1, total // n) if not self.settings.extra_args.strip()
                               else max(1, int(self.settings.threads)))
        if n == 1:
            return [fn(it) for it in items]
        with ThreadPoolExecutor(max_workers=n) as ex:
            return list(ex.map(fn, items))
