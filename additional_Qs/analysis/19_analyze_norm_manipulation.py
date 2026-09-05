#!/usr/bin/env python3
"""Analyze P/C manipulation checks under the locked 0830 edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CONTRACT = "rq3-0830-edit-norming-manipulation-check-v1"
PROMPT_CONTRACT = "rq3-v2-raw-sequence-20260825"
THETAS = (0.25,)
SEED = 20260902
PROJ = Path(__file__).resolve().parents[2]
NORMING = PROJ / "additional_Qs" / "rq3_run_server" / "norming"


def t_summary(values: np.ndarray, alternative: str = "two-sided") -> dict:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    n=len(values); mean=float(values.mean()); sd=float(values.std(ddof=1)); se=sd/np.sqrt(n)
    t=mean/se if se>0 else np.inf*np.sign(mean)
    if alternative == "greater": p=float(stats.t.sf(t,n-1))
    elif alternative == "less": p=float(stats.t.cdf(t,n-1))
    else: p=float(2*stats.t.sf(abs(t),n-1))
    critical=float(stats.t.ppf(.975,n-1))
    return {"n":n,"mean":mean,"sd":sd,"t":float(t),"p":p,
            "ci_lo":mean-critical*se,"ci_hi":mean+critical*se}


def tost(values: np.ndarray, margin: float) -> dict:
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    n=len(values); mean=float(values.mean()); sd=float(values.std(ddof=1)); se=sd/np.sqrt(n)
    if se <= 0:
        p_lower=0.0 if mean>-margin else 1.0
        p_upper=0.0 if mean<margin else 1.0
    else:
        p_lower=float(stats.t.sf((mean+margin)/se,n-1))
        p_upper=float(stats.t.cdf((mean-margin)/se,n-1))
    critical=float(stats.t.ppf(.95,n-1))
    lo90=mean-critical*se; hi90=mean+critical*se
    return {"margin":margin,"tost_p_lower":p_lower,"tost_p_upper":p_upper,
            "tost_p":max(p_lower,p_upper),"ci90_lo":lo90,"ci90_hi":hi90,
            "equivalent":bool(max(p_lower,p_upper)<.05)}


def load_cells(root: Path) -> pd.DataFrame:
    paths=sorted((root/"shards").glob("*/task*/*.csv"))
    if not paths: raise FileNotFoundError(f"No norm manipulation shards under {root}")
    data=pd.concat((pd.read_csv(p) for p in paths),ignore_index=True)
    if set(data.experiment_contract)!={CONTRACT}: raise RuntimeError("Experiment contract mismatch")
    keys=["model_id","task","label","select_frame","eval_frame","target_direction",
          "direction","strength","prefix","hypothesis","metric","variant"]
    if data.duplicated(keys).any(): raise RuntimeError("Duplicate exact norm-manipulation cells")
    return data


def item_effects(data: pd.DataFrame) -> pd.DataFrame:
    group=["model_id","task","label","select_frame","eval_frame","stage",
           "target_direction","direction","control_role","strength","selection_source",
           "metric","prefix","hypothesis"]
    levels=(data.groupby(group,as_index=False)
            .agg(edited=("edited_score","mean"),baseline=("baseline_score","mean"),
                 pc_orientation=("pc_orientation","first")))
    pivot=levels.pivot_table(index=[c for c in group if c not in ("prefix","hypothesis")]+["pc_orientation"],
                             columns=["prefix","hypothesis"],values=["edited","baseline"])
    required=[(v,p,h) for v in ("edited","baseline") for p in ("I","R") for h in ("h2","h3")]
    if any(key not in pivot.columns for key in required): raise RuntimeError("Incomplete I/R x h2/h3 grid")
    pivot=pivot.reset_index()
    flat=[c if isinstance(c,str) else "_".join(x for x in c if x) for c in pivot.columns]
    pivot.columns=flat
    pivot["edited_did"]=(pivot.edited_R_h3-pivot.edited_I_h3)-(pivot.edited_R_h2-pivot.edited_I_h2)
    pivot["baseline_did"]=(pivot.baseline_R_h3-pivot.baseline_I_h3)-(pivot.baseline_R_h2-pivot.baseline_I_h2)
    pivot["delta_did"]=pivot.edited_did-pivot.baseline_did
    # Cross-frame rows are repeated measurements. Average within held item.
    item_keys=["model_id","task","label","stage","target_direction","direction",
               "control_role","strength","selection_source","metric"]
    return (pivot.groupby(item_keys,as_index=False)
            .agg(delta_did=("delta_did","mean"),baseline_did=("baseline_did","mean"),
                 pc_orientation=("pc_orientation","mean"),frames=("eval_frame","nunique")))


def baseline_margins(items: pd.DataFrame) -> pd.DataFrame:
    """Lock margins to all 165 baseline items, not the eligible edit subset."""
    rows=[]
    for model in sorted(items.model_id.unique()):
        slug=model.replace("/","__")
        norm=pd.read_csv(NORMING/f"norm_rq3_{slug}.csv")
        if set(norm.prompt_contract)!={PROMPT_CONTRACT}:
            raise RuntimeError(f"Baseline margin prompt contract mismatch: {model}")
        averaged=(norm.groupby(["label","prefix","hypothesis"],as_index=False)
                  .agg(P=("rho","mean"),C=("C","mean")))
        if averaged.label.nunique()!=165:
            raise RuntimeError(f"Margin anchor must contain 165 items: {model}")
        for metric in ("P","C"):
            wide=averaged.pivot(index="label",columns=["prefix","hypothesis"],values=metric)
            did=((wide[("R","h3")]-wide[("I","h3")])
                 -(wide[("R","h2")]-wide[("I","h2")]))
            anchor=float(np.median(np.abs(did.to_numpy())))
            for theta in THETAS:
                rows.append({"model_id":model,"metric":metric,"theta":theta,
                             "anchor_median_abs_context_did":anchor,"margin":theta*anchor,
                             "n_anchor_items":len(did)})
    return pd.DataFrame(rows)


def analyze(items: pd.DataFrame, margins: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    keys=["model_id","task","stage","target_direction","direction","control_role",
          "strength","selection_source","metric"]
    for key,g in items.groupby(keys):
        meta=dict(zip(keys,key)); values=g.delta_did.to_numpy()
        two=t_summary(values,"two-sided")
        expected=values.copy()
        if meta["target_direction"]=="PC_given_R":
            if meta["metric"]=="C": expected=expected*np.sign(g.pc_orientation.to_numpy())
            if meta["stage"]=="ABL": expected=-expected
            superiority=t_summary(expected,"greater")
        else:
            superiority={"p":np.nan,"mean":np.nan}
        for theta in THETAS:
            margin=float(margins[(margins.model_id==meta["model_id"])
                                 &(margins.metric==meta["metric"])
                                 &(margins.theta==theta)].margin.iloc[0])
            equivalence=tost(values,margin)
            rows.append({**meta,**two,"theta":theta,**equivalence,
                         "expected_oriented_mean":superiority["mean"],
                         "superiority_p":superiority["p"],
                         "superiority_pass":bool(superiority["p"]<.05),
                         "items":g.label.nunique(),"min_frames":int(g.frames.min()),
                         "max_frames":int(g.frames.max())})
    return pd.DataFrame(rows)


def joint_verdict(tests: pd.DataFrame) -> pd.DataFrame:
    target=tests[tests.control_role=="target"].copy()
    keys=["model_id","task","stage","target_direction","strength","selection_source","theta"]
    rows=[]
    for key,g in target.groupby(keys):
        by=g.set_index("metric")
        if set(by.index)!={"P","C"}: raise RuntimeError("Joint verdict needs P and C")
        direction=key[3]
        if direction=="R_given_PC":
            passed=bool(by.equivalent.all()); criterion="P&C TOST equivalence"
            joint_p=float(by.tost_p.max())
        else:
            passed=bool(by.superiority_pass.all()); criterion="P&C oriented superiority"
            joint_p=float(by.superiority_p.max())
        rows.append({**dict(zip(keys,key)),"joint_criterion":criterion,
                     "joint_iut_p":joint_p,"joint_pass":passed,
                     "P_mean":float(by.loc["P","mean"]),"C_mean":float(by.loc["C","mean"]),
                     "P_p":float(by.loc["P","tost_p" if direction=="R_given_PC" else "superiority_p"]),
                     "C_p":float(by.loc["C","tost_p" if direction=="R_given_PC" else "superiority_p"])})
    return pd.DataFrame(rows)


def report(joint: pd.DataFrame, margins: pd.DataFrame, root: Path) -> str:
    primary=joint[joint.theta==.25]
    r_primary=primary[primary.target_direction=="R_given_PC"]
    pc_primary=primary[primary.target_direction=="PC_given_R"]
    lines=["# P/C manipulation check of the 0830 intervention", "",
           "The RQ3-v2 norms P/C were re-measured under the same LOIO direction, layer, seed and ADD/ABL hook.",
           "R|PC passes if both P and C are TOST-equivalent; PC|R passes if both oriented P and C show superiority.",
           "θ is fixed at .25. The joint hypothesis over the two components is an IUT, so the joint p is the maximum of the two raw p-values. Holm-corrected values are neither computed nor used.", "",
           f"- R|PC: P and C both equivalent in {int(r_primary.joint_pass.sum())}/{len(r_primary)} strength cells",
           f"- PC|R: P and C both move in the predicted direction in {int(pc_primary.joint_pass.sum())}/{len(pc_primary)} strength cells", "",
           "## θ=.25: R|PC and PC|R", "",
           "| model | task | stage | direction | strength | source | P Δdid | P raw p | C Δdid | C raw p | joint IUT raw p | pass |",
           "|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in primary.sort_values(["target_direction","model_id","task","strength"]).itertuples():
        lines.append(f"| {r.model_id.split('/')[-1]} | {r.task} | {r.stage} | {r.target_direction} | {r.strength:g} | {r.selection_source} | {r.P_mean:+.4f} | {r.P_p:.4g} | {r.C_mean:+.4f} | {r.C_p:.4g} | {r.joint_iut_p:.4g} | {'yes' if r.joint_pass else 'no'} |")
    lines += ["", "## Equivalence margins", "",
              "δ = per-model median(|baseline context DID|) × θ with θ fixed at .25. The margin is used only in the R|PC TOST, not in the PC|R superiority test.", "",
              margins.round(6).to_markdown(index=False), "",
              "## Interpretive limits", "",
              "The target combinations were chosen after the 0830 results were seen. This check qualifies the interpretation of those positive CF interventions; it is not an independent confirmatory run.",
              "The p-values in the table are the uncorrected p of the preregistered t-based TOST / one-sided superiority tests for P and C. Joint IUT p = max(P p, C p); no Holm correction is applied."]
    return "\n".join(lines)+"\n"


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",type=Path,required=True)
    args=ap.parse_args(); out=args.input_dir/"analysis"; out.mkdir(parents=True,exist_ok=True)
    cells=load_cells(args.input_dir); items=item_effects(cells); margins=baseline_margins(items)
    tests=analyze(items,margins); joint=joint_verdict(tests)
    cells.to_csv(out/"norm_manipulation_cells.csv",index=False)
    items.to_csv(out/"norm_manipulation_item_effects.csv",index=False)
    margins.to_csv(out/"norm_manipulation_margins.csv",index=False)
    tests.to_csv(out/"norm_manipulation_tests.csv",index=False)
    joint.to_csv(out/"norm_manipulation_joint_verdicts.csv",index=False)
    (out/"NORM_MANIPULATION_REPORT.md").write_text(report(joint,margins,args.input_dir),encoding="utf-8")
    manifest={"contract":CONTRACT,"theta":.25,
              "primary":"per-strength P&C IUT without multiplicity correction",
              "multiplicity":"none; joint IUT p=max(P raw p,C raw p)","seed":SEED,
              "shards":len(list((args.input_dir/'shards').glob('*/task*/*.csv'))),
              "rows":len(cells),"item_effect_rows":len(items),"test_rows":len(tests),
              "joint_rows":len(joint)}
    (out/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    primary=joint[joint.theta==.25]
    r=primary[primary.target_direction=="R_given_PC"]
    pc=primary[primary.target_direction=="PC_given_R"]
    print(f"ANALYSIS complete theta=.25 R|PC={r.joint_pass.sum()}/{len(r)} PC|R={pc.joint_pass.sum()}/{len(pc)}")
    return 0


if __name__=="__main__": raise SystemExit(main())
