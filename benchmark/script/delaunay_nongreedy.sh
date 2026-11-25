#!/usr/bin/env bash
# **************************************************************
#  Taskparts counterpart of parlaylib/sandbox/delaunay_scheduler_testing.cpp:
#  measures how "non-greedy" delaunay_nativeforkjoin is, over several
#  runs, for a range of input sizes and worker counts.
#
#  Usage (from taskparts-fork/benchmark):
#    script/delaunay_nongreedy.sh [--procs 4,8,16,32,64] [--nmin 100] [--nmax 1e7]
#                                 [--runs 5] [--out trace_results] [--grid]
#                                 [--window-ns 400] [--bin bin/delaunay_nativeforkjoin.log]
#                                 [--build] [--cxx clang++] [--keep-events]
#                                 [--timeout SECS]
#
#  Appends one row per run to <out>/runs.csv, in the same format as the
#  parlay harness:
#    algo,p,n,run,workers,columns,elapsed_ns,steal_while_ready,min_steal_ready
#  and then, if parlaylib/sandbox/analyze_traces.py is found, writes
#  <out>/summary.csv and <out>/non_greediness.txt from it.
#
#  --grid        also write <out>/raw/delaunay/p<P>/n<n>/run<r>.grid.txt
#                (readable by analyze_traces.py --grid) and run<r>.json
#  --keep-events also keep the raw run<r>.events.txt traces there (large:
#                ~100MB for n=1e5 at 4 workers)
#  --window-ns   sample window; 400 matches parlay's PARLAY_TRACE_SAMPLE_PERIOD_NS
#  --build       build the binary first with `make <bin> CXX=<cxx>`; the
#                default compiler is clang++ because GCC -O3 builds of
#                delaunay_nativeforkjoin crash
#  --timeout     kill a run after SECS seconds (needs `timeout` or `gtimeout`)
#
#  Environment variables (e.g. TASKPARTS_CPU_BASE_FREQUENCY_KHZ, needed on
#  macOS) are passed through to the benchmark.
# **************************************************************
set -euo pipefail

procs="4,8,16,32,64"
nmin="100"
nmax="1e7"
runs=5
out="trace_results"
grid=0
keep_events=0
window_ns=400
bin="bin/delaunay_nativeforkjoin.log"
build=0
cxx="clang++"
timeout_secs=""

usage() { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
need() { [[ $# -ge 2 ]] || { echo "$1 needs a value" >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --procs)       need "$@"; procs="$2"; shift 2 ;;
    --nmin)        need "$@"; nmin="$2"; shift 2 ;;
    --nmax)        need "$@"; nmax="$2"; shift 2 ;;
    --runs)        need "$@"; runs="$2"; shift 2 ;;
    --out)         need "$@"; out="$2"; shift 2 ;;
    --window-ns)   need "$@"; window_ns="$2"; shift 2 ;;
    --bin)         need "$@"; bin="$2"; shift 2 ;;
    --cxx)         need "$@"; cxx="$2"; shift 2 ;;
    --timeout)     need "$@"; timeout_secs="$2"; shift 2 ;;
    --grid)        grid=1; shift ;;
    --keep-events) keep_events=1; shift ;;
    --build)       build=1; shift ;;
    -h|--help)     usage 0 ;;
    *) echo "unknown argument $1" >&2; usage 1 ;;
  esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench_dir="$(cd "$script_dir/.." && pwd)"
nongreedy_py="$script_dir/nongreedy.py"
analyze_py="$bench_dir/../../parlaylib/sandbox/analyze_traces.py"

# accept 1e7-style sizes, like the parlay harness
to_int() { awk -v x="$1" 'BEGIN { printf "%d", x }'; }
nmin=$(to_int "$nmin")
nmax=$(to_int "$nmax")
IFS=',' read -r -a proc_list <<< "$procs"

if [[ $build -eq 1 ]]; then
  (cd "$bench_dir" && make "$bin" CXX="$cxx")
fi
[[ "$bin" = /* ]] || bin="$bench_dir/$bin"
[[ -x "$bin" ]] || { echo "no executable at $bin (build it, or pass --build)" >&2; exit 1; }

timeout_cmd=()
if [[ -n "$timeout_secs" ]]; then
  if command -v timeout >/dev/null; then timeout_cmd=(timeout "$timeout_secs")
  elif command -v gtimeout >/dev/null; then timeout_cmd=(gtimeout "$timeout_secs")
  else echo "warning: --timeout given but no timeout/gtimeout command; running without it" >&2
  fi
fi

mkdir -p "$out"
out="$(cd "$out" && pwd)"
csv="$out/runs.csv"
[[ -f "$csv" ]] || echo "algo,p,n,run,workers,columns,elapsed_ns,steal_while_ready,min_steal_ready" > "$csv"

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/delaunay_nongreedy.XXXXXX")"
trap 'rm -rf "$tmp_root"' EXIT

algo="delaunay"
nb_failed=0
for (( n = nmin; n <= nmax; n *= 10 )); do
  for p in "${proc_list[@]}"; do
    for (( run = 1; run <= runs; run++ )); do
      work="$tmp_root/work"
      rm -rf "$work"; mkdir -p "$work"

      # the benchmark writes log0.json and log0.events.txt to its cwd
      status=0
      (cd "$work" && \
        TASKPARTS_NUM_WORKERS="$p" \
        TASKPARTS_LOGGING_FIBERS=1 \
        TASKPARTS_BENCHMARK_WARMUP_SECS=0 \
        TASKPARTS_BENCHMARK_NUM_REPEAT=1 \
        ${timeout_cmd[@]+"${timeout_cmd[@]}"} "$bin" -n "$n" > stdout.txt 2>&1) || status=$?
      if [[ $status -ne 0 || ! -s "$work/log0.events.txt" ]]; then
        echo "$algo p=$p n=$n run=$run: FAILED (exit status $status); last lines of output:" >&2
        tail -n 5 "$work/stdout.txt" >&2 || true
        nb_failed=$((nb_failed + 1))
        continue
      fi

      raw_dir="$out/raw/$algo/p$p/n$n"
      grid_args=()
      if [[ $grid -eq 1 || $keep_events -eq 1 ]]; then mkdir -p "$raw_dir"; fi
      if [[ $grid -eq 1 ]]; then grid_args=(--grid-out "$raw_dir/run$run.grid.txt"); fi

      run_csv="$work/nongreedy.csv"
      python3 "$nongreedy_py" --window-ns "$window_ns" --csv "$run_csv" \
        ${grid_args[@]+"${grid_args[@]}"} "$work/log0.events.txt" > /dev/null

      # map nongreedy.py's columns (by name) onto the parlay runs.csv format
      row=$(awk -F, -v algo="$algo" -v p="$p" -v n="$n" -v run="$run" '
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", algo, p, n, run,
                 $col["workers"], $col["columns"], $col["elapsed_ns"],
                 $col["steal_while_ready"], $col["min_steal_ready"] }' "$run_csv")
      echo "$row" >> "$csv"
      IFS=',' read -r _ _ _ _ _ cols _ swr msr <<< "$row"
      echo "$algo p=$p n=$n run=$run: columns=$cols S*[ready>0]=$swr min(S,ready)=$msr"

      if [[ $grid -eq 1 ]]; then cp "$work/log0.json" "$raw_dir/run$run.json"; fi
      if [[ $keep_events -eq 1 ]]; then mv "$work/log0.events.txt" "$raw_dir/run$run.events.txt"; fi
    done
  done
done

if [[ -f "$analyze_py" ]]; then
  python3 "$analyze_py" "$csv"
else
  echo "note: $analyze_py not found; skipping summary.csv / non_greediness.txt" >&2
fi
if [[ $nb_failed -gt 0 ]]; then
  echo "$nb_failed run(s) failed" >&2
  exit 1
fi
