#!/usr/bin/env python3
"""Build the RQ3 stimulus file: active 15 items × 3 prefixes.

input : data/stimuli_00_all.csv   (the active 15 items; the single source of truth)
output: data/stimuli_RQ3.csv      (long format, 45 rows = 15 items × {N, I, R})
       data/STIMULI_RQ3.md       (human-readable table with the three prefixes in full)

What is newly created
  1) `org` / `org_irr` / `ctx_irr` for the four Chung items, empty in the source CSV.
     `org_irr` was chosen under the four rules of the RQ3 design document §3.1 and
     passed the mechanical checks (zero word-count difference, no content-word overlap with h1/h2/h3/e).
  2) `QUD` for the four Chung items: a design record, never shown in prompts.
  3) Bare-arm option strings (`opt_b2`, `opt_b3`) with `tgt` as subject.

Design rationale: RQ3 design document §1.1 (three prefixes), §3.0 (materials → hypotheses), §3.1 (generation rules).
"""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(PROJ, "data", "stimuli_00_all.csv")
OUT_CSV = os.path.join(PROJ, "data", "stimuli_RQ3.csv")
OUT_MD = os.path.join(PROJ, "data", "STIMULI_RQ3.md")

# --- supplements for the four Chung items (design document §3.1 candidate table) ------
# org      : the string actually present in ctx; a substitution anchor, so it must be verbatim
# org_irr  : the newly chosen irrelevant ORG; rules 1-4 in the design document §3.1
# qud      : the question the context raises; a design record, never shown in prompts
CHUNG_FILL = {
    "chung-1": dict(
        org="an umbrella company",
        org_irr="a dairy company",
        qud="Does Dan own an umbrella, or does he not?",
        avoided="rain/umbrella (h3) · music/lessons (h2) · violin (e) · hiking (h1)",
    ),
    "chung-2": dict(
        org="a car company",
        org_irr="a mattress company",
        qud="Does Noah have a driving license, or does he not?",
        avoided="driving/cars (h3) · skiing/snow (h2) · Switzerland (e) · baking (h1)",
    ),
    "chung-3": dict(
        org="an American travel agency",
        org_irr="a Belgian chocolate maker",
        qud="Did Carlo spend his summer holidays in America in 2017, or did he not?",
        avoided="America/travel/holiday (h3) · Barcelona/study abroad (h2) · Italy/student (e)",
    ),
    "chung-4": dict(
        org="a shampoo company",
        org_irr="a bicycle company",
        qud="Does Alice have brown hair, or does she not?",
        avoided="hair/beauty (h3) · model/fashion (h2) · Sweden vs furniture/kitchen (e) · studies (h1)",
    ),
}

PREFIXES = [
    ("N", "ctx_none", "no context"),
    ("I", "ctx_irr", "irrelevant context"),
    ("R", "ctx", "relevant context"),
]

STOP = set(
    "a an the is are was were be been being of in on at to for and or not no has have had "
    "he she it his her its with as by from this that who whom does do did there their they "
    "them can could would should will shall may might must if then than so".split()
)


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", str(s).lower()) if w not in STOP}


def bare_option(tgt: str, frag: str) -> str:
    """Bare arm: prepend the subject of h1 to h_i (spec §4.4)."""
    frag = str(frag).strip().rstrip(".")
    return f"{tgt} {frag}."


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(SRC)
    if len(src) != 15:
        sys.exit(f"expected 15 active items, found {len(src)}")

    wide = src.copy()
    wide["org_source"] = "inherited"

    for label, fill in CHUNG_FILL.items():
        m = wide.label == label
        if not m.any():
            sys.exit(f"item not found: {label}")
        i = wide.index[m][0]
        ctx = wide.at[i, "ctx"]
        if fill["org"] not in ctx:
            sys.exit(f"{label}: org anchor not in ctx: {fill['org']!r}")
        wide.at[i, "org"] = fill["org"]
        wide.at[i, "org_irr"] = fill["org_irr"]
        wide.at[i, "ctx_irr"] = ctx.replace(fill["org"], fill["org_irr"])
        wide.at[i, "QUD"] = fill["qud"]
        wide.at[i, "org_source"] = "new-2026-08-24"

    # --- checks ------------------------------------------------------------
    problems = []
    for _, r in wide.iterrows():
        if not isinstance(r.ctx_irr, str) or not r.ctx_irr:
            problems.append(f"{r.label}: ctx_irr missing")
            continue
        if r.ctx_irr == r.ctx:
            problems.append(f"{r.label}: ctx_irr equals ctx")
        if str(r.ctx_none).strip() != str(r.e).strip():
            problems.append(f"{r.label}: ctx_none != e")
        # if org_irr shares content words with h1/h2/h3/e it introduces new lexical overlap
        if isinstance(r.org_irr, str):
            ov = content_words(r.org_irr) & set().union(
                *(content_words(r[f]) for f in ("h1", "h2", "h3", "e"))
            )
            if ov:
                problems.append(f"{r.label}: org_irr content-word overlap {sorted(ov)}")
    if problems:
        sys.exit("checks failed\n  " + "\n  ".join(problems))

    # --- long format -------------------------------------------------------
    rows = []
    for _, r in wide.iterrows():
        opt = dict(
            opt_base=r.opt_base,
            opt_c2=r.opt_h2,
            opt_c3=r.opt_h3,
            opt_b2=bare_option(r.tgt, r.h2),
            opt_b3=bare_option(r.tgt, r.h3),
        )
        for code, col, name in PREFIXES:
            intro = str(r[col]).strip()
            rows.append(
                dict(
                    label=r.label,
                    kind=r.kind,
                    scenario_id=r.scenario_id,
                    paradigm=r.paradigm,
                    tgt=r.tgt,
                    prefix=code,
                    prefix_name=name,
                    intro=intro,
                    intro_words=len(intro.split()),
                    org_used=("" if code == "N" else (r.org if code == "R" else r.org_irr)),
                    org_source=r.org_source,
                    e=r.e,
                    h1=r.h1,
                    h2=r.h2,
                    h3=r.h3,
                    QUD=r.QUD,
                    **opt,
                    norm_source=r.norm_source,
                    P_h2=r.P_h2,
                    P_h3=r.P_h3,
                    C_h2=r.C_h2,
                    C_h3=r.C_h3,
                )
            )
    long = pd.DataFrame(rows)
    return wide, long


def write_md(wide: pd.DataFrame, long: pd.DataFrame) -> None:
    L = []
    A = L.append
    A("# RQ3 stimuli: active 15 items × 3 prefixes\n")
    A("*Generated by `analysis/rq3/make_stimuli_rq3.py`. Design: RQ3 design document §1.1 and §3.1.*")
    A("*The source is `data/stimuli_00_all.csv`; this file is derived and not edited by hand.*\n")
    A("| | |")
    A("|---|---|")
    A(f"| items | {len(wide)} (Chung 4 · edited 3 · model-written 8) |")
    A(f"| prefixes | 3: `N` no context · `I` irrelevant context · `R` relevant context |")
    A(f"| rows (long format) | {len(long)} |")
    A("| options | conjunction `opt_c2`, `opt_c3` · bare `opt_b2`, `opt_b3` · base `opt_base` |")
    A("")
    A("## 1. ORG contrast: `R` and `I` differ only in this noun phrase\n")
    A("| item | `org` (`R`) | `org_irr` (`I`) | source | word diff |")
    A("|---|---|---|---|---:|")
    for _, r in wide.iterrows():
        wR = len(str(r.org).split())
        wI = len(str(r.org_irr).split())
        src = "**new**" if r.org_source == "new-2026-08-24" else "existing"
        A(f"| `{r.label}` | {r.org} | {r.org_irr} | {src} | {wI - wR:+d} |")
    A("")
    A("Word-count difference of the whole prefix (`I` − `R`):\n")
    A("| item | `R` | `I` | diff |")
    A("|---|---:|---:|---:|")
    for lab, g in long[long.prefix.isin(["R", "I"])].groupby("label", sort=False):
        wr = int(g[g.prefix == "R"].intro_words.iloc[0])
        wi = int(g[g.prefix == "I"].intro_words.iloc[0])
        A(f"| `{lab}` | {wr} | {wi} | {wi - wr:+d} |")
    A("")
    A("> Length differences are not controlled: they enter both `c2` and `c3` arms equally and cancel in the DiD")
    A("> (design document §1.1). This table is descriptive evidence that the minimal pair is minimal.\n")
    A("## 2. Items in full\n")
    for _, r in wide.iterrows():
        A(f"### `{r.label}` — {r.scenario_id}\n")
        A("| | |")
        A("|---|---|")
        A(f"| $e$ | {r.e} |")
        A(f"| $h_1$ | {r.h1} |")
        A(f"| $h_2$ (confirmed) | {r.h2} |")
        A(f"| $h_3$ (relevance) | {r.h3} |")
        A(f"| QUD | {r.QUD} |")
        A(f"| norm source | {r.norm_source}: $P$ {r.P_h2}/{r.P_h3} · $C$ {r.C_h2}/{r.C_h3} |")
        A("")
        for code, col, name in PREFIXES:
            A(f"**`{code}` {name}**")
            A("")
            A("> " + str(r[col]).strip().replace("\n", " "))
            A("")
    open(OUT_MD, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    wide, long = build()
    long.to_csv(OUT_CSV, index=False)
    write_md(wide, long)
    n_new = int((wide.org_source == "new-2026-08-24").sum())
    print(f"{len(long)} rows -> {os.path.relpath(OUT_CSV, PROJ)}")
    print(f"          -> {os.path.relpath(OUT_MD, PROJ)}")
    print(f"{n_new} new org_irr · all checks passed")
