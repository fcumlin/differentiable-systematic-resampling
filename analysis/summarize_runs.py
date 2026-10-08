"""Mean and std of the final metrics over the runs of one or more experiments.

Reads <prefix>_<run>/train.log as written by train.py and scripts/train.sh.
Usage (from the repository root):
    python analysis/summarize_runs.py runs/lorenz63/dsr_10db runs/lorenz63/erot_10db
"""
import argparse
import re

import numpy as np

NUM_EPOCHS = 500

# (label, regex for the logged value, scale); stds are sample stds (ddof=1).
METRICS = [
    ('NMSE (dB)', r'Validation NMSE:\s+(\S+)', 1.0),
    ('MSE', r'Test MSE: (\S+)', 1.0),
    ('MSE-5-step', r'Test 5-step MSE: (\S+)', 1.0),
    ('LL', r'Test log likelihood: (\S+)', 1.0),
    ('KLD', r'Test KL divergence: (\S+)', 1.0),
    ('KLD KF (train)', r'Train mean KL\(particles\|\|Kalman\) after resample=([\d.]+)', 1.0),
    ('||A - A_hat||_F', r'Test A matrix distance \(Frobenius\): ([\d.]+)', 1.0),
    ('Time (h)', r'Training completed in ([\d.]+)', 1.0 / 3600),
    ('Time/epoch (s)', r'Training completed in ([\d.]+)', 1.0 / NUM_EPOCHS),
]


def summarize(prefix, num_runs):
    values = {label: [] for label, *_ in METRICS}
    diverged = []
    for i in range(num_runs):
        log_path = f'{prefix}_{i}/train.log'
        try:
            with open(log_path) as f:
                content = f.read()
        except FileNotFoundError:
            print(f'Log file not found: {log_path}')
            continue
        last = {}
        for label, pattern, scale in METRICS:
            found = re.findall(pattern, content)
            if found:
                last[label] = float(found[-1]) * scale
        if 'Training diverged' in content or not all(
            np.isfinite(v) for v in last.values()
        ):
            diverged.append(i)
            continue
        for label, value in last.items():
            values[label].append(value)

    print(f'\n{prefix}')
    if diverged:
        print(f'  diverged runs (left out): {diverged}')
    for label, _, _ in METRICS:
        v = values[label]
        if len(v) > 1:
            print(f'  {label:<18} {np.mean(v):10.4f} +- {np.std(v, ddof=1):.4f}'
                  f'   ({len(v)} runs)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefixes', nargs='+',
                        help='run prefixes, e.g. runs/lorenz63/dsr_10db')
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()
    for prefix in args.prefixes:
        summarize(prefix, args.num_runs)


if __name__ == '__main__':
    main()
