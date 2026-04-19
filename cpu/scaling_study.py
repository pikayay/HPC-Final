#!/usr/bin/env python3
"""
Scaling study for kmeans: runs serial once, then parallel with 1..max_threads.
Captures wall-clock time, iterations to converge, speedup, and efficiency.
Results are written to scaling_results.txt.
"""

import subprocess
import re
import os
import multiprocessing
import argparse
from datetime import datetime

BINARY   = "./kmeans"
INPUT    = "tracks_features_cleaned.csv"
OUTPUT   = "/dev/null"  # discard per-run CSV output
RESULTS  = "scaling_results.txt"


def run_kmeans(mode, threads=None, k=8, max_iters=100, rows=0):
    """Run kmeans and return (elapsed_seconds, iterations, converged, cluster_sizes)."""
    cmd = [
        BINARY,
        "--input",   INPUT,
        "--output",  OUTPUT,
        "--k",       str(k),
        "--iters",   str(max_iters),
        "--mode",    mode,
    ]
    if rows > 0:
        cmd += ["--rows", str(rows)]
    if mode == "parallel" and threads is not None:
        cmd += ["--threads", str(threads)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout

    # Parse elapsed time
    elapsed = None
    m = re.search(r"\[Timing\].*elapsed:\s*([\d.]+)s", stdout)
    if m:
        elapsed = float(m.group(1))

    # Parse iteration count — last "Iter X/" line tells us how many ran
    iters_run = None
    for match in re.finditer(r"Iter (\d+)/\d+", stdout):
        iters_run = int(match.group(1))

    # Converged?
    converged = "Converged" in stdout

    # Cluster sizes
    cluster_sizes = re.findall(r"Cluster \d+:\s*(\d+) songs", stdout)
    cluster_sizes = [int(x) for x in cluster_sizes]

    return elapsed, iters_run, converged, cluster_sizes


def format_row(label, elapsed, iters, converged, speedup, efficiency, cluster_sizes):
    conv_str = "yes" if converged else f"no (hit max)"
    size_str = ", ".join(str(s) for s in cluster_sizes) if cluster_sizes else "N/A"
    return (
        f"  {label:<22}"
        f"  {elapsed:>10.4f}s"
        f"  {iters:>6} iters"
        f"  converged={conv_str:<12}"
        f"  speedup={speedup:>6.3f}x"
        f"  efficiency={efficiency:>6.1f}%"
        f"  clusters=[{size_str}]\n"
    )


def main():
    parser = argparse.ArgumentParser(description="K-Means OpenMP scaling study")
    parser.add_argument("--k",        type=int, default=8,   help="Number of clusters")
    parser.add_argument("--iters",    type=int, default=100, help="Max iterations")
    parser.add_argument("--rows",     type=int, default=0,   help="Max rows (0=all)")
    parser.add_argument("--runs",     type=int, default=3,   help="Runs per config (averaged)")
    parser.add_argument("--max-threads", type=int, default=0,
                        help="Max threads (0 = system CPU count)")
    args = parser.parse_args()

    max_threads = args.max_threads if args.max_threads > 0 else multiprocessing.cpu_count()

    if not os.path.isfile(BINARY):
        print(f"Binary '{BINARY}' not found — run 'make' first.")
        return 1

    print(f"Starting scaling study: k={args.k}, max_iters={args.iters}, "
          f"rows={'all' if args.rows == 0 else args.rows}, "
          f"runs_per_config={args.runs}, max_threads={max_threads}\n")

    def avg_run(mode, threads=None):
        """Return averaged (elapsed, iters, converged, cluster_sizes) over args.runs runs."""
        times, iters_list, convs, sizes = [], [], [], []
        for _ in range(args.runs):
            t, it, cv, cs = run_kmeans(mode, threads, args.k, args.iters, args.rows)
            if t is None:
                print(f"  WARNING: could not parse timing for {mode} threads={threads}")
                return None, None, None, None
            times.append(t)
            iters_list.append(it or 0)
            convs.append(cv)
            sizes = cs  # same each run
        return (sum(times) / len(times),
                round(sum(iters_list) / len(iters_list)),
                all(convs),
                sizes)

    # --- Serial baseline ---
    print("Running serial baseline...")
    serial_time, serial_iters, serial_conv, serial_sizes = avg_run("serial")
    if serial_time is None:
        print("Serial run failed. Aborting.")
        return 1
    print(f"  Serial: {serial_time:.4f}s, {serial_iters} iters\n")

    # --- Parallel 1..max_threads ---
    parallel_results = []
    for t in range(1, max_threads + 1):
        print(f"Running parallel threads={t}/{max_threads}...")
        elapsed, iters, conv, sizes = avg_run("parallel", threads=t)
        if elapsed is None:
            continue
        speedup    = serial_time / elapsed if elapsed > 0 else 0.0
        efficiency = (speedup / t) * 100.0
        parallel_results.append((t, elapsed, iters, conv, speedup, efficiency, sizes))
        print(f"  {elapsed:.4f}s, {iters} iters, speedup={speedup:.3f}x, eff={efficiency:.1f}%")

    # --- Write results file ---
    with open(RESULTS, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("K-MEANS OPENMP SCALING STUDY\n")
        f.write(f"Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Binary:      {os.path.abspath(BINARY)}\n")
        f.write(f"Input:       {INPUT}\n")
        f.write(f"k:           {args.k}\n")
        f.write(f"max_iters:   {args.iters}\n")
        f.write(f"rows:        {'all' if args.rows == 0 else args.rows}\n")
        f.write(f"runs/config: {args.runs} (times averaged)\n")
        f.write(f"max_threads: {max_threads}\n")
        f.write("=" * 100 + "\n\n")

        # Serial
        f.write("SERIAL BASELINE\n")
        f.write("-" * 60 + "\n")
        conv_str  = "yes" if serial_conv else "no (hit max)"
        size_str  = ", ".join(str(s) for s in serial_sizes) if serial_sizes else "N/A"
        f.write(f"  {'serial':<22}  {serial_time:>10.4f}s"
                f"  {serial_iters:>6} iters"
                f"  converged={conv_str:<12}"
                f"  speedup=  1.000x  efficiency= 100.0%"
                f"  clusters=[{size_str}]\n\n")

        # Parallel sweep
        f.write("PARALLEL SCALING (OpenMP)\n")
        f.write("-" * 100 + "\n")
        f.write(f"  {'config':<22}  {'time':>11}  {'iters':>11}  "
                f"{'converged':<22}  {'speedup':<15}  {'efficiency':<16}  clusters\n")
        f.write("-" * 100 + "\n")
        for (t, elapsed, iters, conv, speedup, efficiency, sizes) in parallel_results:
            label = f"parallel (t={t})"
            f.write(format_row(label, elapsed, iters, conv, speedup, efficiency, sizes))

        # Summary table
        f.write("\nSUMMARY TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {'threads':>8}  {'time (s)':>10}  {'speedup':>9}  {'efficiency':>12}\n")
        f.write(f"  {'serial':>8}  {serial_time:>10.4f}  {'1.000x':>9}  {'100.0%':>12}\n")
        for (t, elapsed, iters, conv, speedup, efficiency, _) in parallel_results:
            f.write(f"  {t:>8}  {elapsed:>10.4f}  {speedup:>8.3f}x  {efficiency:>11.1f}%\n")

        # Peak speedup
        if parallel_results:
            best = max(parallel_results, key=lambda r: r[4])
            f.write(f"\nPeak speedup: {best[4]:.3f}x at {best[0]} threads "
                    f"({best[5]:.1f}% efficiency)\n")

    print(f"\nResults written to {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
