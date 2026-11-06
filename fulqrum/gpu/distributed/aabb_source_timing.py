# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


def source_class(same_alpha, same_beta):


    return (
        ("self" if same_beta else "same_alpha_remote_beta")
        if same_alpha
        else ("same_beta_remote_alpha" if same_beta else "mixed")
    )


def rank_summary(rows):

    result = {}
    for key in (
        "kernel_ms",
        "comm_ms",
        "region_ms",
        "aabb_ms",
        "residual_ms",
        "parent_ms",
    ):
        values = [(r["rank"], r[key]) for r in rows if r[key] is not None]
        if not values:
            continue
        low = min(values, key=lambda x: x[1])
        high = max(values, key=lambda x: x[1])
        mean = sum(v for _, v in values) / len(values)
        result[key] = dict(
            min=low[1],
            min_rank=low[0],
            mean=mean,
            max=high[1],
            max_rank=high[0],
            max_mean=high[1] / mean if mean else 0.0,
            ranks=len(values),
        )
    return result


def format_source_timings(iteration, rows):


    def fmt(value):

        return "unavailable" if value is None else f"{value:.3f} ms"

    lines = []
    for row in sorted(rows, key=lambda r: r["rank"]):
        lines.append(
            f"Alpha-beta rank timing: matvec={iteration} rank={row['rank']} "
            + " ".join(
                f"{k}={fmt(row[k])}"
                for k in (
                    "kernel_ms",
                    "comm_ms",
                    "region_ms",
                    "aabb_ms",
                    "residual_ms",
                    "parent_ms",
                )
            )
            + f" covered_sources={row['covered_sources']}/{len(row['sources'])}"
        )
        for src in row["sources"]:
            lines.append(
                f"Alpha-beta source timing: matvec={iteration} rank={row['rank']} source_rank={src['src']} "
                f"class={src['class']} path={src['path']} launches={src['launches']} "
                + " ".join(
                    f"{k}={fmt(src[k])}"
                    for k in ("kernel_ms", "aabb_ms", "residual_ms", "parent_ms")
                )
            )
    for key, stats in rank_summary(rows).items():
        lines.append(
            f"Alpha-beta timing across ranks: matvec={iteration} metric={key} "
            f"min={stats['min']:.3f} ms (rank {stats['min_rank']}) mean={stats['mean']:.3f} ms "
            f"max={stats['max']:.3f} ms (rank {stats['max_rank']}) max_to_mean={stats['max_mean']:.4f} "
            f"measured_ranks={stats['ranks']}/{len(rows)}"
        )
    return lines
