#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate 2D QEC [[N,K,D]] upper-bound tables.

Output CSVs:
1. qec_dmax_singleton.csv
2. qec_dmax_hamming.csv
3. qec_dmax_combined.csv

Rows: K
Columns: N
Cell value: maximum D allowed by the bound

Note:
- Singleton bound applies generally as an upper bound:
    D <= floor((N-K)/2) + 1

- Quantum Hamming bound here is the standard non-degenerate stabilizer bound:
    sum_{j=0}^{t} 3^j * C(N,j) <= 2^(N-K)
  where t = floor((D-1)/2)

- Passing these bounds does not guarantee that the code exists.
"""

import csv
import argparse
from math import comb
from pathlib import Path


def dmax_singleton(N: int, K: int):
    """
    Quantum Singleton bound:
        K <= N - 2(D - 1)
    so
        D <= floor((N-K)/2) + 1
    """
    if K > N:
        return None
    return (N - K) // 2 + 1


def hamming_lhs(N: int, t: int) -> int:
    """
    Compute:
        sum_{j=0}^{t} 3^j * C(N,j)
    """
    return sum((3 ** j) * comb(N, j) for j in range(t + 1))


def dmax_hamming(N: int, K: int):
    """
    Quantum Hamming bound for non-degenerate stabilizer codes:
        sum_{j=0}^{t} 3^j * C(N,j) <= 2^(N-K)

    where:
        t = floor((D-1)/2)

    For the largest valid t_max, the largest D not excluded by
    this bound is:
        D_max = min(N, 2*t_max + 2)
    """
    if K > N:
        return None

    rhs = 2 ** (N - K)
    t_max = 0

    for t in range(0, N + 1):
        lhs = hamming_lhs(N, t)
        if lhs <= rhs:
            t_max = t
        else:
            break

    return min(N, 2 * t_max + 2)


def dmax_combined(N: int, K: int):
    """
    Combined upper bound:
        min(Singleton bound, Hamming bound)
    """
    s = dmax_singleton(N, K)
    h = dmax_hamming(N, K)

    if s is None or h is None:
        return None

    return min(s, h)


def make_table(max_n: int, max_k: int, bound_func):
    """
    Create a 2D table.

    First row:
        K\\N, 1, 2, 3, ..., max_n

    Each following row:
        K, D(N=1), D(N=2), ..., D(N=max_n)
    """
    table = []

    header = ["K\\N"] + list(range(1, max_n + 1))
    table.append(header)

    for K in range(1, max_k + 1):
        row = [K]

        for N in range(1, max_n + 1):
            value = bound_func(N, K)

            if value is None:
                row.append("")
            else:
                row.append(value)

        table.append(row)

    return table


def write_csv(path: Path, table):
    """
    Write table to CSV.
    utf-8-sig helps Excel open Chinese/Unicode cleanly.
    """
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(
        description="Generate QEC [[N,K,D]] Dmax tables as CSV files."
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=100,
        help="Maximum N value. Default: 40",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=100,
        help="Maximum K value. Default: 40",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory. Default: current directory",
    )

    args = parser.parse_args()

    max_n = args.max_n
    max_k = args.max_k
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    singleton_table = make_table(max_n, max_k, dmax_singleton)
    hamming_table = make_table(max_n, max_k, dmax_hamming)
    combined_table = make_table(max_n, max_k, dmax_combined)

    write_csv(out_dir / "qec_dmax_singleton.csv", singleton_table)
    write_csv(out_dir / "qec_dmax_hamming.csv", hamming_table)
    write_csv(out_dir / "qec_dmax_combined.csv", combined_table)

    print("Done. Generated files:")
    print(out_dir / "qec_dmax_singleton.csv")
    print(out_dir / "qec_dmax_hamming.csv")
    print(out_dir / "qec_dmax_combined.csv")


if __name__ == "__main__":
    main()