"""``python -m catdemos warmup``: download the datasets and train/cache every model once."""

from __future__ import annotations

import sys
import time


def warmup() -> None:
    """Download the datasets and train or load every cached model, reporting the time of each."""
    from catdemos.asap.model import load_or_train as asap
    from catdemos.common.adult import Census
    from catdemos.elastic.pipeline import load_or_train as elastic

    for name, step in (
        ("ASAP autoencoders", asap),
        ("Census sample", Census),
        ("Elastic privacy agent", elastic),
    ):
        t0 = time.perf_counter()
        print(f"  {name:<24}", end="", flush=True)
        step()
        print(f"ready ({time.perf_counter() - t0:.1f} s)")


def main() -> None:
    """Command-line entry point (only ``warmup`` is a valid command)."""
    if sys.argv[1:] != ["warmup"]:
        sys.exit("usage: python -m catdemos warmup  (servers: python -m catdemos.<hub|asap|psdc|dp|elastic>)")
    warmup()


if __name__ == "__main__":
    main()
