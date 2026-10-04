"""Train + evaluate a list of experiments, one per GPU in parallel.

    python scripts/run_queue.py --trainval-root ... --test-root ... --out runs \
        D_full B5_authors_baseline A_raw A_raw@1 A_raw@2 D_full@1 D_full@2

Each job is a config name from configs/experiments/. "<config>@<k>" is an extra
random seed k, saved as run "<config>_s<k>" and evaluated without maps.
Finished runs (test/test_metrics.csv present) are skipped. Runs listed in --mc
also get an MC-dropout evaluation (uncertainty maps). Per-run logs go to
<out>/<run>/train.log and eval.log; progress lines are echoed with a [gpuN] prefix.
"""
import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ("epoch", "best ", "early stop", "Error", "error", "Traceback", "ALL", "done")


def parse(job):
    cfg, _, seed = job.partition("@")
    name = f"{cfg}_s{seed}" if seed else cfg
    over = [f"train.seed={seed}", f"name={name}"] if seed else []
    return cfg, name, over, not seed  # maps only for the main (seed 42) run


def stream(cmd, env, log, tag):
    """Run cmd, write everything to log, echo progress lines to stdout."""
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            if any(k in line for k in PROGRESS):
                print(f"[{tag}] {line.rstrip()}", flush=True)
        return p.wait()


def worker(gpu, jobs, args, results):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tag = f"gpu{gpu}" if gpu is not None else "cpu"
    py = sys.executable
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        cfg, name, over, maps = parse(job)
        run = Path(args.out) / name
        t0 = time.time()
        if (run / "test" / "test_metrics.csv").exists():
            print(f"[{tag}] skip {name}: already done", flush=True)
        else:
            run.mkdir(parents=True, exist_ok=True)
            print(f"[{tag}] start {name}", flush=True)
            dev = ["--device", args.device] if args.device else []
            rc = stream([py, "scripts/train.py", "--config", f"configs/experiments/{cfg}.yaml",
                         "--data-root", args.trainval_root, "--out", args.out, *dev,
                         f"data.num_workers={args.workers}", *args.extra, *over],
                        env, run / "train.log", f"{tag} {name}")
            if rc == 0:
                rc = stream([py, "scripts/evaluate.py", "--run", str(run), "--test-root", args.test_root,
                             "--mc", "0", "--workers", str(args.workers), *dev, *([] if maps else ["--no-maps"])],
                            env, run / "eval.log", f"{tag} {name}")
            if rc != 0:
                print(f"[{tag}] FAILED {name} (exit {rc}); see {run}/train.log and eval.log", flush=True)
                results[name] = "failed"
                continue
        if name in args.mc and not (run / "test_mc" / "test_metrics.csv").exists():
            rc = stream([py, "scripts/evaluate.py", "--run", str(run), "--test-root", args.test_root,
                         "--mc", "20", "--tag", "mc", "--workers", str(args.workers),
                         *(["--device", args.device] if args.device else [])],
                        env, run / "eval_mc.log", f"{tag} {name} mc")
            if rc != 0:
                print(f"[{tag}] FAILED {name} MC evaluation; see {run}/eval_mc.log", flush=True)
                results[name] = "failed"
                continue
        results[name] = "ok"
        print(f"[{tag}] finished {name} in {(time.time() - t0) / 60:.0f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--trainval-root", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--mc", nargs="*", default=["D_full"], help="runs that also get MC-dropout evaluation")
    ap.add_argument("--gpus", type=int, default=None, help="parallel workers (default: number of GPUs)")
    ap.add_argument("--workers", type=int, default=None, help="data-loader workers per run")
    ap.add_argument("--device", help="force a device (e.g. cpu) for local tests")
    ap.add_argument("--extra", nargs="*", default=[], help="config overrides for every run")
    args = ap.parse_args()

    for j in args.jobs:  # fail fast on typos before any GPU time is spent
        cfg = parse(j)[0]
        if not (ROOT / "configs" / "experiments" / f"{cfg}.yaml").exists():
            sys.exit(f"unknown experiment {cfg!r}")
    for p in (args.trainval_root, args.test_root):
        if not Path(p).exists():
            sys.exit(f"missing input folder {p}")

    import torch
    n_gpu = torch.cuda.device_count() if args.device is None else 0
    slots = args.gpus or max(1, n_gpu)
    gpus = list(range(slots)) if n_gpu else [None] * slots
    if args.workers is None:
        args.workers = max(2, (os.cpu_count() or 4) // slots + 1)
    print(f"{len(args.jobs)} jobs on {slots} worker(s) {gpus}, {args.workers} data workers each", flush=True)

    jobs = queue.Queue()
    for j in args.jobs:
        jobs.put(j)
    results = {}
    threads = [threading.Thread(target=worker, args=(g, jobs, args, results)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\nsummary:", ", ".join(f"{k}={v}" for k, v in results.items()), flush=True)
    if any(v != "ok" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
