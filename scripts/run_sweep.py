"""Resumable sweep runner. Run from a terminal (not a notebook kernel):

    .venv\\Scripts\\python.exe scripts/run_sweep.py configs/main_addition.yaml
    .venv\\Scripts\\python.exe scripts/run_sweep.py configs/main_addition.yaml --dry

Each config in the expanded grid is trained (resuming from checkpoints/<run_id>/last.pt if a
previous attempt was interrupted) and evaluated; finished runs (results/runs/<run_id>/metrics.json)
are skipped. One status line per run is appended to results/sweeps/<config name>.jsonl.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glwm2 import RESULTS_DIR  # noqa: E402
from glwm2.train.pipeline import run  # noqa: E402
from glwm2.train.trainer import run_dirs  # noqa: E402
from glwm2.utils.config import expand_grid, load_yaml  # noqa: E402
from glwm2.utils.io import append_jsonl  # noqa: E402


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def claim(rid_dir: Path) -> bool:
    """Per-run lock so several sweep processes can share one grid. Stale locks (dead pid) are taken over."""
    rid_dir.mkdir(parents=True, exist_ok=True)
    lock = rid_dir / "sweep.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = -1
        if pid != os.getpid() and pid > 0 and _pid_alive(pid):
            return False
        lock.unlink(missing_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--dry", action="store_true", help="list runs and their status only")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--posthoc-iters", type=int, default=1500)
    ap.add_argument("--reverse", action="store_true", help="walk the grid backwards (second worker)")
    args = ap.parse_args()

    for cfg_path in (args.configs[::-1] if args.reverse else args.configs):
        spec = load_yaml(cfg_path)
        cfgs = expand_grid(spec)[:: -1 if args.reverse else 1]
        name = Path(cfg_path).stem
        log = RESULTS_DIR / "sweeps" / f"{name}.jsonl"
        done = [c for c in cfgs if (run_dirs(c)[1] / "metrics.json").exists()]
        print(f"{name}: {len(cfgs)} runs, {len(done)} finished")
        if args.dry:
            for c in cfgs:
                print(("  done  " if c in done else "  todo  ") + c.run_id())
            continue
        n = 0
        for c in cfgs:
            if c in done:
                continue
            if args.max_runs is not None and n >= args.max_runs:
                break
            if (run_dirs(c)[1] / "metrics.json").exists() or not claim(run_dirs(c)[0]):
                continue
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] start {c.run_id()}", flush=True)
            try:
                run(c, posthoc_iters=args.posthoc_iters)
                status = "ok"
            except KeyboardInterrupt:
                append_jsonl(log, {"run_id": c.run_id(), "status": "interrupted", "sec": time.time() - t0})
                raise
            except Exception as exc:  # keep the sweep going; the traceback goes to the log
                status = "error: " + repr(exc)
                traceback.print_exc()
            append_jsonl(log, {"run_id": c.run_id(), "status": status, "sec": round(time.time() - t0, 1),
                               "config": c.to_dict()})
            print(f"[{time.strftime('%H:%M:%S')}] {status} {c.run_id()} ({time.time() - t0:.0f}s)", flush=True)
            (run_dirs(c)[0] / "sweep.lock").unlink(missing_ok=True)
            if "CUDA" in status:
                # the CUDA context is unusable after an OOM / device error; stop instead of failing every run
                sys.exit(3)
            n += 1


if __name__ == "__main__":
    main()
