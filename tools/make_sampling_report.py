#!/usr/bin/env python3
"""Build docs/sampling.md from the measurement records.

Usage: make_sampling_report.py <repository root>

Reads docs/sampling/records.json (written by bench/sampling_report.cpp), draws
the figures into docs/sampling/, and fills docs/sampling.md.in into
docs/sampling.md. Every number in the report comes from the records; the
script stops if the template asks for a value it does not compute.
"""

import json
import math
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Reference palette, first four categorical slots (validated light mode).
SERIES = {
    "rank": ("#2a78d6", "o", "rank"),
    "coordinate_scrambled": ("#eb6834", "s", "coordinate (scrambled)"),
    "iid": ("#1baf7a", "^", "iid"),
    "scaled_rank": ("#eda100", "D", "scaled rank"),
}
# Short names for the labels at line ends; the legend carries the full names.
SHORT = {"rank": "rank", "coordinate_scrambled": "coordinate", "iid": "iid", "scaled_rank": "scaled"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SPACE_TITLES = {
    "small": "6 parts",
    "medium": "8 parts",
    "near_cap": "30 parts, total near the cap sum",
    "wide": "100 parts (195-bit size)",
}
TABLE_N = (100, 1024, 1100, 16384)
ALL_SEQUENCES = ("rank", "rank_scrambled", "coordinate", "coordinate_scrambled", "iid", "scaled_rank")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def label_ends(fig, ax, lines, min_gap_pt=9.0):
    """Direct labels at the right end of each line, spread vertically so that
    no two are closer than min_gap_pt. Call after the layout is final."""
    fig.canvas.draw()
    to_pt = 72.0 / fig.dpi
    ends = []
    for ln in lines:
        x, y = ln.get_xdata()[-1], ln.get_ydata()[-1]
        ends.append([ax.transData.transform((x, y))[1] * to_pt, ln, x, y])
    ends.sort(key=lambda e: e[0])
    placed = [e[0] for e in ends]
    for i in range(1, len(placed)):  # push up from the bottom
        placed[i] = max(placed[i], placed[i - 1] + min_gap_pt)
    shift = (sum(e[0] for e in ends) - sum(placed)) / len(placed)  # recentre
    for (orig, ln, x, y), p in zip(ends, placed):
        ax.annotate(ln.get_label(), xy=(x, y), xytext=(6, p + shift - orig),
                    textcoords="offset points", va="center", fontsize=7, color=TEXT)


def series_rows(records, space, seq, key):
    rows = sorted((r for r in records if r["space"] == space and r["sequence"] == seq),
                  key=lambda r: r["n"])
    return [r["n"] for r in rows], [r[key] for r in rows]


def figure_by_space(records, key, ylabel, path, log_y, reference=None):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharey=True, facecolor=SURFACE)
    to_label = []
    for ax, space in zip(axes.flat, SPACE_TITLES):
        style(ax)
        lines = []
        for seq, (color, marker, name) in SERIES.items():
            x, y = series_rows(records, space, seq, key)
            (ln,) = ax.plot(x, y, color=color, marker=marker, markersize=3.5, linewidth=1.4, label=SHORT[seq])
            lines.append(ln)
        if reference:
            fn, _ = reference
            x, _ = series_rows(records, space, "rank", key)
            ax.plot(x, [fn(n) for n in x], color=MUTED, linewidth=0.8, label="_ref")
        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_title(SPACE_TITLES[space], fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("points N", fontsize=8, color=MUTED)
        ax.set_xlim(right=ax.get_xlim()[1] * 4)
        to_label.append((ax, lines))
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    handles = [Line2D([], [], color=c, marker=m, linewidth=1.4, markersize=4) for c, m, _ in SERIES.values()]
    names = [n for _, _, n in SERIES.values()]
    if reference:
        handles.append(Line2D([], [], color=MUTED, linewidth=0.8))
        names.append(reference[1])
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ax, lines in to_label:
        label_ends(fig, ax, lines)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_by_part(records, path, space, n):
    fig, ax = plt.subplots(figsize=(9.5, 3.4), facecolor=SURFACE)
    style(ax)
    lines = []
    for seq in ("rank", "coordinate_scrambled", "iid"):
        color, _, _ = SERIES[seq]
        name = SHORT[seq]
        row = next(r for r in records if r["space"] == space and r["sequence"] == seq and r["n"] == n)
        parts = list(range(len(row["marginal_by_part"])))
        (ln,) = ax.plot(parts, row["marginal_by_part"], color=color, linewidth=1.2, label=name)
        lines.append(ln)
    ax.set_xlabel("part index", fontsize=8, color=MUTED)
    ax.set_ylabel("largest marginal error", fontsize=8, color=MUTED)
    ax.set_xlim(right=len(parts) * 1.12)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    label_ends(fig, ax, lines)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def figure_threads(records, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.6), facecolor=SURFACE)
    style(ax)
    lines = []
    for seq in ("rank", "coordinate_scrambled", "iid"):
        color, marker, _ = SERIES[seq]
        name = SHORT[seq]
        rows = sorted((r for r in records if r["sequence"] == seq), key=lambda r: r["threads"])
        (ln,) = ax.plot([r["threads"] for r in rows], [r["points_per_second"] / 1e6 for r in rows],
                        color=color, marker=marker, markersize=4, linewidth=1.4, label=name)
        lines.append(ln)
    counts = sorted({r["threads"] for r in records})
    ax.set_xscale("log", base=2)
    ax.set_xticks(counts)
    ax.set_xticklabels([str(t) for t in counts])
    ax.minorticks_off()
    ax.set_xlabel("threads", fontsize=8, color=MUTED)
    ax.set_ylabel("million points per second", fontsize=8, color=MUTED)
    ax.set_xlim(counts[0] / 1.3, counts[-1] * 1.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    label_ends(fig, ax, lines)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


ACCEPT_SERIES = (("rank_scrambled", "rank"), ("coordinate_scrambled", "coordinate_scrambled"), ("iid", "iid"))


def acceptance_rms(acc):
    """Root-mean-square error of the acceptance estimates over every subtree
    and seed, by sequence and number of draws; and the binomial value that
    independent draws should give."""
    subs = acc["subtrees"]
    exact = [s["accepted"] / s["size"] for s in subs]
    rms = {}
    for seq, _ in ACCEPT_SERIES:
        for di, d in enumerate(acc["draws"]):
            err = [reps[di] / d - e for s, e in zip(subs, exact) for reps in s["estimates"][seq]]
            if len(err) != len(subs) * acc["seeds"]:
                raise SystemExit("acceptance records miss replicates")
            rms[(seq, d)] = math.sqrt(sum(x * x for x in err) / len(err))
    binomial = {d: math.sqrt(sum(e * (1 - e) / d for e in exact) / len(exact)) for d in acc["draws"]}
    return rms, binomial


def figure_acceptance(acc, path):
    """Left: exact acceptance of each subtree. Right: error of its estimates
    against the number of draws."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.4), facecolor=SURFACE)
    subs = acc["subtrees"]
    style(left)
    left.plot([s["prefix"][0] for s in subs], [s["accepted"] / s["size"] for s in subs],
              color=MUTED, marker="o", markersize=3.5, linewidth=1.4)
    left.set_title("exact acceptance of each subtree", fontsize=9, color=TEXT, loc="left")
    left.set_xlabel("value of the first part", fontsize=8, color=MUTED)
    left.set_ylabel("acceptance", fontsize=8, color=MUTED)
    left.set_ylim(0, 1.02)
    left.xaxis.set_major_locator(MaxNLocator(integer=True))
    style(right)
    rms, binomial = acceptance_rms(acc)
    lines = []
    for seq, key in ACCEPT_SERIES:
        color, marker, _ = SERIES[key]
        (ln,) = right.plot(acc["draws"], [rms[(seq, d)] for d in acc["draws"]], color=color, marker=marker,
                           markersize=3.5, linewidth=1.4, label=SHORT[key])
        lines.append(ln)
    right.plot(acc["draws"], [binomial[d] for d in acc["draws"]], color=MUTED, linewidth=0.8)
    right.set_xscale("log", base=2)
    right.set_yscale("log")
    right.set_xticks(acc["draws"])
    right.set_xticklabels([str(d) for d in acc["draws"]])
    right.minorticks_off()
    right.set_xlim(right=acc["draws"][-1] * 2.2)
    right.set_title("error of estimates from the first points", fontsize=9, color=TEXT, loc="left")
    right.set_xlabel("points drawn per subtree", fontsize=8, color=MUTED)
    right.set_ylabel("root-mean-square error", fontsize=8, color=MUTED)
    handles = [Line2D([], [], color=SERIES[k][0], marker=SERIES[k][1], linewidth=1.4, markersize=4)
               for _, k in ACCEPT_SERIES] + [Line2D([], [], color=MUTED, linewidth=0.8)]
    names = ["rank (scrambled)", "coordinate (scrambled)", "iid", "iid, expected"]
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    label_ends(fig, right, lines)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def describe_groups(groups, parts):
    """Words for a list of group limits: runs of adjacent parts with the same
    size and sums are counted together."""
    runs, out = {}, []
    for g in groups:
        ps = sorted(g["parts"])
        sums = f"{g['min_sum']}..{g['max_sum']}"
        if ps == list(range(ps[0], ps[-1] + 1)):
            runs.setdefault((len(ps), sums), []).append((ps[0], ps[-1]))
        elif ps == list(range(0, parts, 2)):
            out.append(f"the even parts summing to {sums}")
        else:
            out.append("parts " + ", ".join(map(str, ps)) + f" summing to {sums}")
    words = []
    for (size, sums), spans in runs.items():
        if len(spans) == 1:
            words.append(f"parts {spans[0][0]} to {spans[0][1]} summing to {sums}")
        else:
            words.append(f"{len(spans)} groups of {size} adjacent parts, each summing to {sums}")
    return words + out


def describe_limits(sp):
    words = []
    if sp["min_count"] > 0 or sp["max_count"] < sp["parts"]:
        words.append(f"{sp['min_count']} to {sp['max_count']} nonzero parts")
    if sp["min_size"]:
        if len(set(sp["min_size"])) != 1:
            raise SystemExit("describe_limits handles one min_size for every part")
        words.append(f"each part 0 or at least {sp['min_size'][0]}")
    words += describe_groups(sp["groups"], sp["parts"])
    return "; ".join(words) if words else "none"


def fmt(x, digits=3):
    if isinstance(x, int):
        return f"{x:,}".replace(",", " ")
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-5:
        return f"{x:.{digits - 1}e}"
    if abs(x) >= 10 ** digits:
        return f"{x:.0f}"
    return f"{x:#.{digits}g}".rstrip(".")


def spread_table(records, space):
    head = "| sequence | " + " | ".join(f"N = {n}" for n in TABLE_N) + " |"
    sep = "|---|" + "---:|" * len(TABLE_N)
    out = []
    for key, title in (("gap_ratio", "Largest rank gap times N / size"),
                       ("marginal_max", "Largest marginal error (any part, any value)"),
                       ("distinct", "Distinct points")):
        out += [f"*{title}*", "", head, sep]
        for seq in ALL_SEQUENCES:
            cells = []
            for n in TABLE_N:
                row = next(r for r in records if r["space"] == space and r["sequence"] == seq and r["n"] == n)
                cells.append(fmt(row[key]) if key == "distinct" else fmt(float(row[key])))
            out.append(f"| {seq} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out).rstrip()


def main():
    if len(sys.argv) != 2:
        print((__doc__ or "").strip())
        return 2
    root = pathlib.Path(sys.argv[1])
    folder = root / "docs" / "sampling"
    data = json.loads((folder / "records.json").read_text())
    spread = data["spread"]
    spaces = {s["name"]: s for s in data["spaces"]}

    figure_by_space(spread, "gap_ratio", "largest rank gap x N / size", folder / "gap_ratio.png", log_y=False)
    figure_by_space(spread, "marginal_max", "largest marginal error", folder / "marginal.png", log_y=True,
                    reference=(lambda n: 1.0 / math.sqrt(n), "1/sqrt(N)"))
    figure_by_part(spread, folder / "by_part.png", "wide", max(r["n"] for r in spread))
    figure_threads(data["throughput"], folder / "threads.png")

    def rows(seq, space=None, n=None):
        return [r for r in spread if r["sequence"] == seq and (space is None or r["space"] == space)
                and (n is None or r["n"] == n)]

    def worst(seq, key="gap_ratio"):
        return max(r[key] for r in rows(seq))

    def pow2(n):
        return n & (n - 1) == 0

    values = {}
    m = data["machine"]
    values["machine"] = (f"{m['cpu']}, {m['hardware_threads']} hardware threads, {m['compiler']}, "
                         f"{'optimised (release)' if m['optimised'] else 'unoptimised'} build, "
                         f"library {m['library_version']}")
    for name, s in spaces.items():
        values[f"size_{name}"] = s["size"] if s["size_bits"] <= 40 else f"{int(s['size']):.3e}"
        values[f"bits_{name}"] = str(s["size_bits"])
        lo, hi = s["lower"], s["upper"]
        if len(set(lo)) == 1 and len(set(hi)) == 1:
            values[f"bounds_{name}"] = f"each part {lo[0]}..{hi[0]}"
        else:
            values[f"bounds_{name}"] = "parts " + ", ".join(f"{a}..{b}" for a, b in zip(lo, hi))
        values[f"total_{name}"] = str(s["total"])
        values[f"parts_{name}"] = str(s["parts"])
    for seq in ALL_SEQUENCES:
        values[f"worst_gap_{seq}"] = fmt(worst(seq))
    def two_m_over_size(sp):
        size = int(spaces[sp]["size"])
        m = max(1, (size - 1).bit_length())
        return (1 << m) / size
    lines = ["| space | measured at N = 1024 | 2^m / size |", "|---|---:|---:|"]
    for sp in SPACE_TITLES:
        measured = float(next(r["gap_ratio"] for r in rows("rank", sp, 1024)))
        lines.append(f"| {SPACE_TITLES[sp]} | {fmt(measured)} | {fmt(two_m_over_size(sp))} |")
    values["table_rank_pow2"] = "\n".join(lines)
    values["scaled_gap_pow2_max"] = fmt(max(r["gap_ratio"] for r in rows("scaled_rank") if pow2(r["n"]) and r["n"] >= 64))
    pairs = list(zip(sorted(rows("rank"), key=lambda r: (r["space"], r["n"])),
                     sorted(rows("rank_scrambled"), key=lambda r: (r["space"], r["n"]))))
    if any(x["space"] != y["space"] or x["n"] != y["n"] for x, y in pairs):
        raise SystemExit("rank and rank_scrambled records do not line up")
    values["rank_same_gap"] = str(sum(1 for x, y in pairs if x["gap_ratio"] == y["gap_ratio"]))
    values["rank_gap_cases"] = str(len(pairs))
    for sp in SPACE_TITLES:
        values[f"table_{sp}"] = spread_table(spread, sp)
    top_n = max(r["n"] for r in spread)
    wide = {seq: next(r for r in rows(seq, "wide", top_n)) for seq in ("rank", "coordinate_scrambled", "iid")}
    values["wide_marg_rank"] = fmt(wide["rank"]["marginal_max"])
    values["wide_marg_coord"] = fmt(wide["coordinate_scrambled"]["marginal_max"])
    values["wide_marg_iid"] = fmt(wide["iid"]["marginal_max"])
    ranges = ((0, 10, "0 to 9"), (10, 50, "10 to 49"), (50, 100, "50 to 99"))
    lines = ["| parts | rank | coordinate (scrambled) | iid |", "|---|---:|---:|---:|"]
    for a, b_, label in ranges:
        cells = []
        for seq in ("rank", "coordinate_scrambled", "iid"):
            part = wide[seq]["marginal_by_part"][a:b_]
            cells.append(fmt(sum(part) / len(part)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    values["table_by_part"] = "\n".join(lines)
    first10 = lambda seq: max(wide[seq]["marginal_by_part"][:10])  # noqa: E731
    last50 = lambda seq: max(wide[seq]["marginal_by_part"][50:])   # noqa: E731
    values["wide_first10_rank"] = fmt(first10("rank"))
    values["wide_last50_rank"] = fmt(last50("rank"))
    values["wide_first10_iid"] = fmt(first10("iid"))
    values["wide_last50_iid"] = fmt(last50("iid"))
    values["wide_last50_coord"] = fmt(last50("coordinate_scrambled"))
    dup = {seq: min(r["distinct"] / r["n"] for r in rows(seq, "small", top_n)) for seq in ("coordinate", "coordinate_scrambled", "iid")}
    values["small_distinct_coord"] = f"{dup['coordinate_scrambled']:.1%}"
    values["small_distinct_iid"] = f"{dup['iid']:.1%}"
    values["small_size_fraction"] = f"{top_n / int(spaces['small']['size']):.0%}"

    res = data["resume"]
    if any(r["first"] != res[0]["first"] or r["per_region"] != res[0]["per_region"] for r in res):
        raise SystemExit("resume records use different settings")
    values["resume_first"] = str(res[0]["first"])
    values["resume_per_region"] = str(res[0]["per_region"])
    lines = ["| space | subtrees | points added | repeats | worst gap ratio inside a subtree |",
             "|---|---:|---:|---:|---:|"]
    for sp in SPACE_TITLES:
        r = next(x for x in res if x["space"] == sp)
        lines.append(f"| {SPACE_TITLES[sp]} | {r['regions']} | {r['added']} | {r['repeats']} | "
                     f"{fmt(r['worst_gap_ratio'])} |")
    values["table_resume"] = "\n".join(lines)
    fr = data["fresh"]
    keys = ("prefix", "region_size", "first", "visited_inside")
    if any(r[k] != fr[0][k] for r in fr for k in keys):
        raise SystemExit("fresh records use different settings")
    values["fresh_prefix"] = "(" + ", ".join(str(v) for v in fr[0]["prefix"]) + ")"
    values["fresh_size"] = str(fr[0]["region_size"])
    values["fresh_first"] = str(fr[0]["first"])
    values["fresh_inside"] = str(fr[0]["visited_inside"])
    lines = ["| source | points handed out | visited points passed over | moved to rank order |",
             "|---|---:|---:|---|"]
    for r in fr:
        lines.append(f"| {r['source']} | {r['handed_out']} | {r['skipped']} | "
                     f"{'yes' if r['switched'] else 'no'} |")
    values["table_fresh"] = "\n".join(lines)

    # Section 6: constrained spaces.
    lim = data["limited_spaces"]
    base = lim[0]
    if base["name"] != "base" or describe_limits(base) != "none":
        raise SystemExit("the first limited space should be the unconstrained base")
    if any(s["parts"] != base["parts"] or s["total"] != base["total"] or s["lower"] != base["lower"]
           or s["upper"] != base["upper"] for s in lim):
        raise SystemExit("limited spaces do not share one base")
    if len(set(base["lower"])) != 1 or len(set(base["upper"])) != 1 or base["lower"][0] != 0:
        raise SystemExit("the base should bound every part by 0..cap")
    values["limited_parts"] = str(base["parts"])
    values["limited_total"] = str(base["total"])
    values["limited_cap"] = str(base["upper"][0])
    if any(s["builds"] != base["builds"] for s in lim):
        raise SystemExit("limited spaces were built a different number of times")
    values["limited_build_runs"] = {3: "three", 5: "five"}.get(base["builds"], str(base["builds"]))
    ops = {(r["space"], r["op"]): r["ns"] for r in data["limited_ops"]}
    lines = ["| space | limits | size | layers per row, at most | table MiB | build ms | "
             "logarithmic table MiB | build ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in lim:
        lines.append(f"| {s['name']} | {describe_limits(s)} | {int(s['size']):.2e} | {fmt(s['layers'])} | "
                     f"{fmt(s['table_bytes'] / 2**20)} | {fmt(s['build_seconds'] * 1e3)} | "
                     f"{fmt(s['log_table_bytes'] / 2**20)} | {fmt(s['log_build_seconds'] * 1e3)} |")
    values["table_limited"] = "\n".join(lines)
    lines = ["| space | unrank | rank | decode | decode (logarithmic) | repair |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in lim:
        cells = [fmt(ops[(s["name"], op)] / 1000) for op in ("unrank", "rank", "decode_exact", "decode_log", "repair")]
        lines.append(f"| {s['name']} | " + " | ".join(cells) + " |")
    values["table_limited_ops"] = "\n".join(lines)
    by_name = {s["name"]: s for s in lim}
    for name in ("count", "groups", "nested", "crossing", "combined"):
        values[f"layers_{name}"] = fmt(by_name[name]["layers"])
        values[f"mib_{name}"] = fmt(by_name[name]["table_bytes"] / 2**20)
        values[f"ms_build_{name}"] = fmt(by_name[name]["build_seconds"] * 1e3)
    values["count_range"] = str(by_name["count"]["max_count"] + 1)
    if by_name["count"]["layers"] != by_name["count"]["max_count"] + 1:
        raise SystemExit("the count space's layers are not its count range")
    values["total_plus_one"] = str(base["total"] + 1)
    unrank_ratio = max(ops[(s["name"], "unrank")] / ops[("base", "unrank")] for s in lim)
    values["unrank_ratio_max"] = f"{unrank_ratio:.1f}"
    repairs = sorted((ops[(s["name"], "repair")], s["name"]) for s in lim if s["name"] != "base")
    values["us_repair_base"] = fmt(ops[("base", "repair")] / 1000)
    values["ms_repair_min"] = fmt(repairs[0][0] / 1e6)
    values["repair_min_name"] = repairs[0][1]
    values["ms_repair_max"] = fmt(repairs[-1][0] / 1e6)
    values["repair_max_name"] = repairs[-1][1]

    lsp = data["limited_spread"]
    top_lim = max(r["n"] for r in lsp)
    values["limited_n"] = str(top_lim)
    lim_seqs = ("rank", "rank_scrambled", "coordinate_scrambled", "iid")
    for key, title in (("gap_ratio", "gap"), ("marginal_max", "marginal")):
        lines = ["| space | " + " | ".join(lim_seqs) + " |", "|---|" + "---:|" * len(lim_seqs)]
        for s in lim:
            cells = [fmt(float(next(r[key] for r in lsp if r["space"] == s["name"] and r["sequence"] == q
                                    and r["n"] == top_lim))) for q in lim_seqs]
            lines.append(f"| {s['name']} | " + " | ".join(cells) + " |")
        values[f"table_limited_{title}"] = "\n".join(lines)
    values["limited_worst_gap_rank"] = fmt(max(r["gap_ratio"] for r in lsp if r["sequence"].startswith("rank")))
    # The first part's largest error in points (error times N) at powers of two.
    first_part = max(r["marginal_by_part"][0] * r["n"] for r in lsp
                     if r["sequence"] == "coordinate_scrambled" and pow2(r["n"]))
    if first_part > 2:
        raise SystemExit("coordinate first part is more than two points off at a power of two")
    values["limited_first_part_points"] = f"{first_part:.2f}"
    # The prose says the coordinate sequence has the smallest marginal error on every space.
    for s in lim:
        at = {r["sequence"]: r["marginal_max"] for r in lsp if r["space"] == s["name"] and r["n"] == top_lim}
        if any(at["coordinate_scrambled"] >= at[q] for q in lim_seqs if q != "coordinate_scrambled"):
            raise SystemExit(f"coordinate is not the smallest marginal error on {s['name']}")

    acc = data["acceptance"]
    subs = acc["subtrees"]
    if any(len(s["prefix"]) != 1 for s in subs):
        raise SystemExit("acceptance subtrees are not one level down")
    exact = [s["accepted"] / s["size"] for s in subs]
    values["accept_filter"] = (" + ".join(f"{w} x[{i}]" if w != 1 else f"x[{i}]" for i, w in enumerate(acc["weights"]))
                               + f" <= {acc['limit']}")
    values["accept_whole"] = f"{sum(s['accepted'] for s in subs) / sum(s['size'] for s in subs):.1%}"
    values["accept_min"] = f"{min(exact):.1%}"
    values["accept_max"] = f"{max(exact):.1%}"
    values["accept_subtrees"] = str(len(subs))
    values["accept_space_parts"] = str(spaces[acc["space"]]["parts"])
    values["accept_seeds"] = str(acc["seeds"])
    figure_acceptance(acc, folder / "acceptance.png")
    rms, binomial = acceptance_rms(acc)
    shown = [d for d in acc["draws"] if d in (16, 64, 256, 1024)]
    lines = ["| sequence | " + " | ".join(f"{d} points" for d in shown) + " |", "|---|" + "---:|" * len(shown)]
    for seq, _ in ACCEPT_SERIES:
        lines.append(f"| {seq} | " + " | ".join(fmt(rms[(seq, d)]) for d in shown) + " |")
    lines.append("| iid, expected | " + " | ".join(fmt(binomial[d]) for d in shown) + " |")
    values["table_acceptance"] = "\n".join(lines)
    top_d = acc["draws"][-1]
    values["accept_top_draws"] = str(top_d)
    values["accept_rank_gain"] = f"{rms[('iid', top_d)] / rms[('rank_scrambled', top_d)]:.1f}"
    values["accept_coord_gain"] = f"{rms[('iid', top_d)] / rms[('coordinate_scrambled', top_d)]:.1f}"
    # The prose says independent draws follow the binomial error.
    if any(abs(rms[("iid", d)] / binomial[d] - 1) > 0.2 for d in acc["draws"]):
        raise SystemExit("iid acceptance error is off its binomial value")

    ff = data["fresh_filter"]
    keys = ("prefix", "region_size", "first", "visited_inside", "new_accepted", "new_rejected", "weights", "limit")
    if any(r[k] != ff[0][k] for r in ff for k in keys):
        raise SystemExit("fresh_filter records use different settings")
    f0 = ff[0]
    values["ff_filter"] = (" + ".join(f"{w} x[{i}]" if w != 1 else f"x[{i}]" for i, w in enumerate(f0["weights"]))
                           + f" <= {f0['limit']}")
    values["ff_prefix"] = "(" + ", ".join(str(v) for v in f0["prefix"]) + ")"
    values["ff_size"] = str(f0["region_size"])
    values["ff_first"] = str(f0["first"])
    values["ff_inside"] = str(f0["visited_inside"])
    values["ff_new_accepted"] = str(f0["new_accepted"])
    values["ff_new_rejected"] = str(f0["new_rejected"])
    lines = ["| source | points handed out | rejected | visited points passed over | moved to rank order |",
             "|---|---:|---:|---:|---|"]
    for r in ff:
        lines.append(f"| {r['source']} | {r['handed_out']} | {r['rejected']} | {r['skipped']} | "
                     f"{'yes' if r['switched'] else 'no'} |")
    values["table_fresh_filter"] = "\n".join(lines)

    ns = sorted({r["n"] for r in spread})
    values["n_checkpoints"] = str(len(ns))
    values["max_n"] = str(ns[-1])
    thr = data["throughput"]
    values["throughput_points"] = fmt(thr[0]["points"])
    values["throughput_runs"] = str(thr[0]["runs"])
    if any(r["points"] != thr[0]["points"] or r["runs"] != thr[0]["runs"] for r in thr):
        raise SystemExit("throughput records use different settings")
    def pps(seq, t):
        return next(r["points_per_second"] for r in thr if r["sequence"] == seq and r["threads"] == t)
    tmax = max(r["threads"] for r in thr)
    values["threads_max"] = str(tmax)
    lines = ["| sequence | " + " | ".join(f"{t} thread{'s' if t > 1 else ''}" for t in sorted({r['threads'] for r in thr})) + " |",
             "|---|" + "---:|" * len({r['threads'] for r in thr})]
    for seq in ("rank", "coordinate_scrambled", "iid"):
        lines.append(f"| {seq} | " + " | ".join(fmt(pps(seq, t) / 1e6) for t in sorted({r['threads'] for r in thr})) + " |")
    values["table_threads"] = "\n".join(lines)
    values["speedup_8_rank"] = f"{pps('rank', 8) / pps('rank', 1):.1f}"
    values["speedup_max_rank"] = f"{pps('rank', tmax) / pps('rank', 1):.1f}"
    values["pps_1_rank"] = fmt(pps("rank", 1) / 1e6)

    b = data["build"]
    lines = ["| counting | parts | total | upper bound | build ms | table MiB | log size |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in b:
        lines.append(f"| {r['mode']} | {r['parts']} | {r['total']} | {r['cap']} | {fmt(r['seconds'] * 1e3)} | "
                     f"{fmt(r['bytes'] / 2**20)} | {fmt(r['log_size'])} |")
    values["table_build"] = "\n".join(lines)
    exact_1000 = next(r for r in b if r["mode"] == "exact" and r["parts"] == 1000)
    log_1000 = next(r for r in b if r["mode"] == "logarithmic" and r["parts"] == 1000)
    values["mib_exact_1000"] = fmt(exact_1000["bytes"] / 2**20)
    values["mib_log_1000"] = fmt(log_1000["bytes"] / 2**20)

    lat = {r["op"]: r["ns"] for r in data["latency"]}
    lines = ["| operation | microseconds per call |", "|---|---:|"]
    for op in ("unrank", "rank", "decode_exact", "decode_log", "encode_exact", "encode_log"):
        lines.append(f"| {op} | {fmt(lat[op] / 1000)} |")
    values["table_latency"] = "\n".join(lines)
    values["us_unrank"] = fmt(lat["unrank"] / 1000)
    values["us_encode_exact"] = fmt(lat["encode_exact"] / 1000)
    values["us_encode_log"] = fmt(lat["encode_log"] / 1000)

    template = (root / "docs" / "sampling.md.in").read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", template)) - values.keys())
    if missing:
        print("template asks for values the records do not give: " + ", ".join(missing))
        return 1
    report = re.sub(r"\{\{(\w+)\}\}", lambda mm: values[mm.group(1)], template)
    header = "<!-- Generated by tools/make_sampling_report.py from docs/sampling.md.in and docs/sampling/records.json. Edit the template, not this file. -->\n\n"
    (root / "docs" / "sampling.md").write_text(header + report)
    print("wrote docs/sampling.md and figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
