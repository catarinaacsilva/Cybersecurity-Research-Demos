#!/usr/bin/env bash
# Set up and start the demos: ./run.sh [all|hub|asap|psdc|dp|elastic ...]
#
#   - finds Python >= 3.11 (newest first; macOS /usr/bin/python3 is often 3.9)
#   - creates ./venv and installs the dependencies (again only when pyproject.toml changes)
#   - downloads the datasets and trains the models once (cached in ./data)
#   - starts every server, waits until each answers /health, prints the URLs
#   - Ctrl-C stops them all
#
# Environment: HOST (default 127.0.0.1; use 0.0.0.0 to serve the LAN), CATDEMOS_DATA (default ./data),
#              CATDEMOS_THREADS (native math threads per server, default 2).
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"

HOST="${HOST:-127.0.0.1}"
export CATDEMOS_DATA="${CATDEMOS_DATA:-$PWD/data}"
# Native math threads per server (CATDEMOS_THREADS, default 2). The models are small, and five
# servers each starting one thread per core oversubscribe the CPU (measured: a DP point takes
# 26 ms with 2 threads, 67 ms with 16). Set for the servers only, whatever the shell exports.
threads="${CATDEMOS_THREADS:-2}"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
  VECLIB_MAXIMUM_THREADS="$threads"
# Plain functions, not associative arrays: macOS still ships bash 3.2.
port_of() { case $1 in hub) echo 8000 ;; asap) echo 8001 ;; psdc) echo 8002 ;; dp) echo 8003 ;; elastic) echo 8004 ;; *) return 1 ;; esac; }
name_of() {
  case $1 in hub) echo "Start page" ;; asap) echo "ASAP Latent Map" ;; psdc) echo "PsDC Categorizer" ;;
    dp) echo "DP vs Profiling" ;; elastic) echo "Elastic Privacy Map" ;; esac
}
lan_ip() { ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true; }

log() { printf '\033[1;38;2;255;129;69m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

if [[ $# -eq 0 || $1 == all ]]; then targets=(hub asap psdc dp elastic); else targets=("$@"); fi
for t in "${targets[@]}"; do port_of "$t" >/dev/null || die "unknown demo '$t' (hub, asap, psdc, dp, elastic)"; done

# --- Python + venv -------------------------------------------------------------------
if [[ ! -x venv/bin/python ]]; then
  PY=""
  for c in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$c"; break; fi
  done
  [[ -n $PY ]] || die "Python >= 3.11 not found (macOS: brew install python@3.12)"
  log "creating venv with $($PY --version)"
  "$PY" -m venv venv
fi
stamp="venv/.deps-$(cksum pyproject.toml | cut -d' ' -f1)"
if [[ ! -f $stamp ]]; then
  log "installing dependencies (first run takes a minute)"
  venv/bin/python -m pip install -q --upgrade pip
  venv/bin/python -m pip install -q .
  rm -f venv/.deps-* && touch "$stamp"
fi

# --- data + models ----------------------------------------------------------------------
log "preparing datasets and models (first run downloads ~6 MB and trains for ~1 min)"
PYTHONPATH=src venv/bin/python -W ignore -m catdemos warmup

# --- servers ------------------------------------------------------------------------------
pids=""
cleanup() {
  trap - INT TERM EXIT
  log "stopping"
  [[ -z $pids ]] || kill $pids 2>/dev/null || true  # a server that already exited must not skip the wait
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

mkdir -p "$CATDEMOS_DATA/logs"
healthy() { venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$1/health', timeout=1)" 2>/dev/null; }

for t in "${targets[@]}"; do
  p=$(port_of "$t")
  if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then die "port $p is already in use ($(name_of "$t"))"; fi
  PYTHONPATH=src venv/bin/python -W ignore -m "catdemos.$t" --host "$HOST" --port "$p" >"$CATDEMOS_DATA/logs/$t.log" 2>&1 &
  pids="$pids $!"
done

for t in "${targets[@]}"; do
  p=$(port_of "$t")
  for _ in $(seq 120); do healthy "$p" && break; sleep 0.5; done
  healthy "$p" || die "$(name_of "$t") did not start, see $CATDEMOS_DATA/logs/$t.log"
done

show="$HOST"
if [[ $HOST == 0.0.0.0 ]]; then show="$(lan_ip)"; show="${show:-localhost}"; fi
echo
for t in "${targets[@]}"; do printf '  %-20s \033[1mhttp://%s:%s/\033[0m\n' "$(name_of "$t")" "$show" "$(port_of "$t")"; done
echo
log "open the first URL above; Ctrl-C stops everything (logs in $CATDEMOS_DATA/logs)"
wait
