#!/usr/bin/env python3
"""Non-greediness of a taskparts run, in the style of parlaylib's
PARLAY_TRACE_STATES measure.

Input: the <base>.events.txt file written by Logging::output when the
benchmark is built with -DTASKPARTS_LOG and run with
TASKPARTS_LOGGING_FIBERS=1 (e.g., log0.events.txt).

Worker states are rebuilt from the scheduler's wait/sleep events:
  S  between enter_wait and exit_wait (in acquire, i.e., stealing)
  A  between enter_sleep and exit_sleep (elastic only)
  D  after worker_exit
  W  otherwise
The number of ready tasks R is the running sum of fiber_ready (+1) and
fiber_start (-1) over all workers. Both are sampled on a grid of windows,
and with S = #workers in state S in a column:
  steal_while_ready = sum of S * [R > 0]
  min_steal_ready   = sum of min(S, R)

The window length defaults to the mean steal time (mean duration of
enter_wait..exit_wait) measured from the same log; override it with
--window-ticks or --window-ns (e.g., --window-ns 400 to match parlay).

Usage: nongreedy.py [--window-ticks N | --window-ns X] [--grid-out PATH]
                    [--csv PATH] events.txt...
"""
import argparse
import os
import sys

STATE_OF_EVENT = {
    "enter_wait": "S",
    "exit_wait": "W",
    "enter_sleep": "A",
    "exit_sleep": "S",
    "worker_exit": "D",
}


def load(path):
    ns_per_tick, nb_workers = None, None
    events = []  # (ticks, worker, name), already sorted by Logging::output
    with open(path) as f:
        for line in f:
            if line.startswith("# ns_per_tick"):
                ns_per_tick = float(line.split()[2])
            elif line.startswith("# nb_workers"):
                nb_workers = int(line.split()[2])
            elif line.startswith("#") or line.startswith("worker"):
                continue
            else:
                w, t, name = line.split()
                events.append((int(t), int(w), name))
    events.sort(key=lambda e: e[0])
    if nb_workers is None:
        nb_workers = 1 + max((w for _, w, _ in events), default=0)
    return ns_per_tick, nb_workers, events


def mean_steal_ticks(events, nb_workers):
    start = [None] * nb_workers
    total, count = 0, 0
    for t, w, name in events:
        if name == "enter_wait":
            start[w] = t
        elif name == "exit_wait" and start[w] is not None:
            total += t - start[w]
            count += 1
            start[w] = None
    return (total / count) if count > 0 else None


def check_balance(path, events):
    ready = 0
    for t, w, name in events:
        if name == "fiber_ready":
            ready += 1
        elif name == "fiber_start":
            ready -= 1
            if ready < 0:
                print(f"warning: {path}: ready count went negative at ticks={t} "
                      f"(worker {w}); a fiber_ready is missing", file=sys.stderr)
                return
    if ready != 0:
        print(f"warning: {path}: {ready} fiber(s) still ready at end of log", file=sys.stderr)


def summarize(events, nb_workers, window, want_grid):
    end = events[-1][0] if events else 0
    nb_columns = int(end // window) + 1
    state = ["W"] * nb_workers
    nb_stealing = 0
    ready = 0
    swr, msr = 0, 0
    rows = [[] for _ in range(nb_workers)] if want_grid else None
    ready_col = [] if want_grid else None
    e = 0
    for k in range(nb_columns):
        t = k * window
        while e < len(events) and events[e][0] <= t:
            _, w, name = events[e]
            e += 1
            if name == "fiber_ready":
                ready += 1
            elif name == "fiber_start":
                ready -= 1
            elif name in STATE_OF_EVENT:
                s = STATE_OF_EVENT[name]
                nb_stealing += (s == "S") - (state[w] == "S")
                state[w] = s
        swr += nb_stealing * (ready > 0)
        msr += min(nb_stealing, max(ready, 0))
        if want_grid:
            for w in range(nb_workers):
                rows[w].append(state[w])
            ready_col.append(ready)
    grid = (["".join(r) for r in rows], ready_col) if want_grid else None
    return nb_columns, end, swr, msr, grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+", help="<base>.events.txt file(s)")
    win = ap.add_mutually_exclusive_group()
    win.add_argument("--window-ticks", type=float, help="window length in cycle-counter ticks")
    win.add_argument("--window-ns", type=float, help="window length in ns (uses the ns_per_tick header)")
    ap.add_argument("--grid-out", help="write a parlay-style grid (one events file only); "
                    "readable by parlaylib/sandbox/analyze_traces.py --grid")
    ap.add_argument("--csv", help="append one row per events file to this CSV")
    args = ap.parse_args()
    if args.grid_out and len(args.events) != 1:
        sys.exit("--grid-out takes exactly one events file")

    header = ("file,workers,window_ticks,window_ns,columns,elapsed_ns,"
              "steal_while_ready,min_steal_ready,rel_steal_while_ready,rel_min_steal_ready")
    print(f"{'file':<32} {'workers':>7} {'window ns':>10} {'columns':>10} "
          f"{'S*[R>0]':>10} {'min(S,R)':>10} {'S*[R>0]/col':>12} {'min(S,R)/col':>13}")
    out_rows = []
    for path in args.events:
        ns_per_tick, nb_workers, events = load(path)
        if args.window_ticks is not None:
            window = args.window_ticks
        elif args.window_ns is not None:
            if not ns_per_tick:
                sys.exit(f"{path}: no ns_per_tick header; use --window-ticks")
            window = args.window_ns / ns_per_tick
        else:
            window = mean_steal_ticks(events, nb_workers)
            if window is None:
                print(f"skipping {path}: no steals in log to set the window; "
                      f"pass --window-ticks or --window-ns", file=sys.stderr)
                continue
        if window <= 0:
            sys.exit(f"{path}: window must be positive")
        check_balance(path, events)
        cols, end, swr, msr, grid = summarize(events, nb_workers, window, args.grid_out is not None)
        window_ns = window * ns_per_tick if ns_per_tick else float("nan")
        elapsed_ns = end * ns_per_tick if ns_per_tick else float("nan")
        print(f"{os.path.relpath(path):<32} {nb_workers:>7} {window_ns:>10.1f} {cols:>10} "
              f"{swr:>10} {msr:>10} {swr / cols:>12.4f} {msr / cols:>13.4f}")
        out_rows.append(f"{path},{nb_workers},{window:.3f},{window_ns:.3f},{cols},{elapsed_ns:.0f},"
                        f"{swr},{msr},{swr / cols:.6f},{msr / cols:.6f}")
        if grid is not None:
            rows, ready_col = grid
            with open(args.grid_out, "w") as f:
                for r in rows:
                    f.write(r + "\n")
                f.write("\n" + " ".join(str(max(v, 0)) for v in ready_col) + "\n")

    if args.csv:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a") as f:
            if new:
                f.write(header + "\n")
            for r in out_rows:
                f.write(r + "\n")


if __name__ == "__main__":
    main()
