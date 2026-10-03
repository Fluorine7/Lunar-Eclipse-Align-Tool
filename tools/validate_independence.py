"""Real-image order/worker invariance audit; reads images, writes no images.

Run: python tools/validate_independence.py sample_da --radius 370 430
Use --limit for a smaller audit. Reference geometry is deliberately absent.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import iter_frame_analyses
from utils_common import SUPPORTED_EXTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--radius', type=float, nargs=2, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    files = sorted(str(p) for p in args.input.iterdir() if p.is_file()
                   and not p.name.startswith('._') and p.suffix.lower() in SUPPORTED_EXTS)
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error('No supported images')
    params = (*args.radius, 50, 30)
    baseline = None
    for workers, order in ((1, files), (2, files[::-1]),
                           (4, [files[i] for i in np.random.default_rng(123).permutation(len(files))])):
        started = time.perf_counter()
        results = {}
        for path, loaded, error in iter_frame_analyses(order, params, workers):
            if error:
                raise error
            fit = loaded[1].geometry
            results[path] = asdict(fit) if fit else None
        if baseline is None:
            baseline = results
        elif results != baseline:
            differences = [Path(name).name for name in files if results[name] != baseline[name]]
            raise AssertionError(f'Order/worker dependence: {differences}')
        print(json.dumps({'workers': workers, 'frames': len(files),
                          'analysis_seconds': time.perf_counter() - started,
                          'valid': sum(f is not None and not f['reasons'] for f in results.values()),
                          'geometry_exactly_equal': results == baseline}), flush=True)
    # One-image-at-a-time mode with cold geometry-template caches is tested in
    # unit tests; this verifies selected real files in isolation as well.
    for i in np.unique(np.linspace(0, len(files)-1, min(10, len(files))).astype(int)):
        path, loaded, error = next(iter_frame_analyses([files[i]], params, 1))
        if error:
            raise error
        actual = asdict(loaded[1].geometry) if loaded[1].geometry else None
        assert actual == baseline[path], path
    print('Single-image comparisons: PASS', flush=True)


if __name__ == '__main__':
    main()
