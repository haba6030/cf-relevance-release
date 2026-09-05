#!/usr/bin/env python3
"""Mechanical checks on item candidates: the gate to pass before the judge calls.

    python3 data/llm-build/checks/auto_checks.py data/llm-build/items/candidates/*.json
    python3 data/llm-build/checks/auto_checks.py --existing        # check the nine items of sim/scenarios9.py

No dependencies. If `transformers` is available, length-match is re-checked with real tokenizers (recommended).

Passing the machine gate does not mean "this item is valid" but **"it is fit to be sent to humans and
judges"** (proposal §2.12.6 note). Complementarity, logical independence and propositional identity are not here,
because they cannot be checked mechanically.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "sim"))
import chung_frame  # noqa: E402  -- single source of the Chung context carrier

# ------------------------------------------------------------ two tiers of rules
#
# the checker is **a tool for building items consistent with prior work**, not for auditing
# prior work. Hence two tiers.
#
#   tier 1 GATE: applies to every item. Only rules that the Chung et al. (2024) stimuli **actually satisfy 4/4**
#              go here: unique-item · domain-cap · qud-implicit · control-length · no-name-leak · carrier-intact/org-slot · no-complement · evidence-present.
#              the justification is the practice of the prior work itself.
#
#   tier 2 GEN_ONLY: generation constraints applied **to newly written items only**.
#              length-match ratio · no-overlap/overlap-symmetric lexical overlap · org-no-overlap (ORG-slot overlap).
#              Chung stimuli are replication targets and are out of scope, not exempt:
#              editing them would end the replication.
#              out-of-scope findings are still printed, tagged `[out of scope, reference]`.
GEN_ONLY = {"length-match", "length-match-w", "no-overlap", "overlap-symmetric", "org-no-overlap"}


def provenance(it: dict) -> str:
    """Provenance for scoping. ctx_status == 'chung' is a replication target and outside GEN_ONLY."""
    if it.get("provenance"):
        return it["provenance"]
    if it.get("ctx_status") == "chung":
        return "chung"
    return "new"


def out_of_scope(it: dict) -> set:
    """The set of rules that do **not** apply to this item."""
    return GEN_ONLY if provenance(it) == "chung" else set()


# tokenizers used for checking, chosen to cover the model families under test.
TOKENIZERS = ["Qwen/Qwen3-1.7B", "meta-llama/Llama-3.1-8B", "microsoft/Phi-4-mini-instruct"]

STOP = set("""
a an the this that these those there here and or but if then than as of to in on at by for with
from into over under about after before during between within without is are was were be been being
am do does did doing have has had having will would shall should can could may might must
he she it they them his her its their who whom whose which what when where why how
i you we us our your my me him not no nor so such own same too very just also only
one two three four five randomly selected person people someone something
""".split())

# the Tentori Exp1 family uses h3 = not-h2 (h3 is the complement of h2); no-complement catches it.
def is_complement_design(h2: str, h3: str) -> bool:
    return re.sub(r"\bnot\s+", "", h3 or "", flags=re.I).strip().lower() == (h2 or "").strip().lower()

# content words of the carrier itself; overlap with h2/h3 is caught by no-overlap.
# they must also be listed as forbidden words in the prompt (prompts/10_generate_item.md).
CARRIER_WORDS = ("ask call company consultant discover europe group market new "
                 "question research start target try").split()



# ------------------------------------------------------------------ lexical tools
def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text or "")


def stem(w: str) -> str:
    """A very simple stemmer.

    Note: an earlier version used `rstrip("'s")`, which strips a **character set**:
    `compass` became `compa` and false-matched the carrier's `company` by prefix.
    Only the possessive should be cut.
    """
    w = re.sub(r"'s$", "", w.lower())
    if w.endswith("ss"):                  # compass, glass, class ...
        return w
    for suf in ("ies", "ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content(text: str, names: set[str]) -> set[str]:
    """Content-word set, excluding stopwords and registered proper names (capitalised)."""
    low = {n.lower().rstrip(".") for n in names}
    out = set()
    for w in words(text):
        wl = w.lower()
        if wl in STOP or wl in low or len(wl) < 3:
            continue
        out.add(stem(wl))
    return out


def overlap(a: set[str], b: set[str]) -> set[str]:
    """Overlapping content words by stem match plus prefix match (5+ characters).

    Without the prefix rule `America` / `American` pass as different words,
    which is exactly the case in Chung's italian-undergrad item.
    """
    hits = set(a & b)
    for x in a:
        for y in b:
            if x == y:
                continue
            lo, hi = sorted((x, y), key=len)
            if len(lo) >= 5 and hi.startswith(lo):
                hits.add(lo)
    return hits


def known_names(it: dict) -> set[str]:
    """Names excluded from overlap counting: only the target and the consultant.

    A "capitalised = proper noun" rule would exclude nationality adjectives such as
    `American`, `Swiss`, `Italian` wholesale and miss, in Chung's italian-undergrad,
    the overlap between ORG `an American travel agency` and h3 `... in America ...`.
    """
    names = {it.get("tgt", "")}
    ctx = it.get("ctx", "")
    if it.get("consultant"):
        names.add(it["consultant"])
    elif ctx:
        m = re.match(r"\s*([A-Z][a-z]+)", ctx)   # first word of the carrier = CONSULTANT slot
        if m:
            names.add(m.group(1))
    return {n for n in names if n}


def syllables(w: str) -> int:
    """Approximate English syllable count; enough for checking minimal-pair length balance."""
    w = w.lower().strip()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1)


def ntok(text: str) -> int:
    return len(words(text))


def ntok_g1(text: str) -> int:
    """Word count for length-match (conjunct length). **Hyphens are split.**

    Subword tokenizers usually split `right-handed` into 3 pieces. Counting on whitespace alone
    gives 1, so `is right-handed` (2) vs `is a man` (3) looks like a violation, whereas
    Qwen3-1.7B and Phi-4-mini give 3 for both. Treating hyphens as boundaries matches them.
    """
    return len(re.findall(r"[A-Za-z][A-Za-z']*", text or ""))


# ------------------------------------------------------------------ item checks
def check_item(it: dict, corpus: dict) -> list[tuple[str, str]]:
    """List of (rule_id, message) violations; empty means pass."""
    bad: list[tuple[str, str]] = []

    def fail(rule, msg):
        bad.append((rule, msg))

    h2, h3, ctx = it.get("h2", ""), it.get("h3", ""), it.get("ctx", "")
    names = known_names(it)

    # length-match.  The normalised DV is lp/n, so a context effect scales as 1/n2 vs 1/n3.
    # the further the token-count ratio of the two conjuncts is from 1, the more that asymmetry alone moves the DV.
    n2, n3 = ntok_g1(h2), ntok_g1(h3)
    if h2 and h3:
        if abs(n2 - n3) > 1:
            fail("length-match", f"h2/h3 token-count difference {abs(n2-n3)} (>1): {n2} vs {n3}")
        if min(n2, n3) < 3:
            fail("length-match", f"conjuncts too short ({n2}, {n3}); at least 3 tokens are needed to match the length ratio")
        elif max(n2, n3) / min(n2, n3) > 1.34:
            fail("length-match", f"h2/h3 token-count ratio {max(n2,n3)/min(n2,n3):.2f} (>1.34): normalisation-scale asymmetry")
        lo, hi = sorted((len(h2), len(h3)))
        if hi > lo * 1.25:
            fail("length-match-w", f"h2/h3 character-length difference {hi/lo-1:.0%} (>25%, warning)")

    # no-overlap lexical overlap: highest priority
    #
    # e appears in **both** conditions (no context = e alone; context = carrier ending in e), so
    # priming from e cancels in the difference. What does not cancel is the part present only under context,
    # i.e. **carrier + ORG**. Hence ctx minus e is checked, not the whole ctx.
    ctx_only = ctx.replace(it.get("e") or "\0", " ")
    c_ctx = content(ctx_only, names)
    ov2 = overlap(c_ctx, content(h2, names))
    ov3 = overlap(c_ctx, content(h3, names))
    if ov3:
        fail("no-overlap", f"ctx ∩ h3 = {sorted(ov3)}  <- lexical priming; rewrite (P-RELEX) needed")
    if ov2:
        fail("no-overlap", f"ctx ∩ h2 = {sorted(ov2)}")

    # overlap-symmetric
    if len(ov2) != len(ov3):
        fail("overlap-symmetric", f"asymmetric overlap: |ctx∩h2|={len(ov2)}, |ctx∩h3|={len(ov3)}")

    # no-complement (from the audit).  If h3 = not h2 the two are **two cells of the same question**, so
    #     no context can make h3 alone relevant, in principle.
    #     earlier versions **exempted** this form from the negation ban, passing rather than catching it.
    if is_complement_design(h2, h3):
        fail("no-complement", f"complement pair: h3 is the negation of h2 ({h2!r} / {h3!r}). "
                    f"the answerhood contrast fails; h3 must be rewritten as an independent proposition")

    # evidence-present: M-A needs evidence e.  A-B (no e) mismatches the Chung carrier and the no-context
    #     condition in information content, so it is flagged as a design deviation.
    par = it.get("paradigm", "MA")
    if par == "MA" and not (it.get("e") or "").strip():
        fail("evidence-present", "paradigm=MA but e is empty")
    if par == "AB":
        fail("evidence-present-w", "A-B paradigm (no e). The no-context condition is the single line 'Consider a randomly selected "
                     "person, X.', not comparable in information to M-A items (warning)")

    n = ntok(ctx)

    # carrier-intact: Chung carrier inheritance.  ctx must be slot filling, not free writing.
    for msg in chung_frame.conforms(ctx, it.get("paradigm", "MA")):
        fail("carrier-intact", msg)

    # org-slot: the ORG slot, the only place a no-overlap violation can arise
    org = chung_frame.extract_org(ctx)
    if org is None:
        if not any(r == "carrier-intact" for r, _ in bad):
            fail("org-slot", "cannot recover the ORG slot")
    else:
        c_org = content(org, names)
        for label, hyp in (("h2", h2), ("h3", h3)):
            hit = overlap(c_org, content(hyp, names))
            if hit:
                fail("org-no-overlap", f"ORG slot {org!r} shares content words with {label}: {sorted(hit)}")
        if it.get("org") and it["org"].strip() != org.strip():
            fail("org-slot", f"org field ({it['org']!r}) disagrees with the slot recovered from ctx ({org!r})")

    # qud-implicit: Q must not appear verbatim in ctx
    q = (it.get("Q") or "").strip().rstrip("?").lower()
    if q and q[:40] and q[:40] in ctx.lower():
        fail("qud-implicit", "Q appears verbatim in ctx; the context must raise the QUD implicitly")

    # control-length: ctx_irr length match
    if it.get("ctx_irr"):
        ni = ntok(it["ctx_irr"])
        if n and abs(ni - n) > 0.10 * n:
            fail("control-length", f"ctx_irr length {ni} vs ctx {n} (>10% apart)")
        if overlap(content(it["ctx_irr"], names), content(h2, names) | content(h3, names)):
            fail("control-length", "ctx_irr shares content words with h2/h3")

    # no-name-leak
    tgt = (it.get("tgt") or "").lower().rstrip(".")
    if tgt and tgt in (h2 + " " + h3).lower():
        fail("no-name-leak", "target name appears in h2/h3")

    # unique-item global uniqueness · domain-cap
    #
    # catches **h3-only duplicates**, not just pairs. Same h3 means the same relevance target;
    # even with a different ORG the same semantic area is measured twice and the items are not independent.
    # (e.g. chung-4 `has brown hair` + ling-fix-3 `has brown hair`: the pair check alone passed
    #  because h2 differed.)
    pair = (h2.strip().lower(), h3.strip().lower())
    if corpus["pairs"][pair] > 1:
        fail("unique-item", f"(h2,h3) pair duplicated ×{corpus['pairs'][pair]}: {pair}")
    for f, lab in (("h2", "h2"), ("h3", "h3")):
        v = (it.get(f) or "").strip().lower()
        if v and corpus[f][v] > 1:
            fail("unique-item", f"{lab} duplicated ×{corpus[f][v]}: {v!r}; "
                       f"same relevance-manipulation target")
    for f in ("e", "h1"):
        v = (it.get(f) or "").strip().lower()
        if v and corpus[f][v] > 1:
            fail("unique-item", f"{f} duplicated ×{corpus[f][v]}")
    dom = (it.get("domain") or "").strip().lower()
    if dom and corpus["domain"][dom] > 3:
        fail("domain-cap", f"domain {dom!r} has {corpus['domain'][dom]} items (>3)")

    # norm bands: **all warnings** (audit B-2)
    #
    # P_*/C_* are self-reported values written by the generator. What is checked here is
    # "did the model write a number within the band", not whether the item has the property.
    # the real verdict comes only from measured norming.
    try:
        if float(it["C_h2"]) < 2.0:
            fail("norm-confirms-w", f"declared C_h2={it['C_h2']} < 2.0 (self-report, not measured)")
        if not -1.0 <= float(it["C_h3"]) <= 1.0:
            fail("norm-inert-w", f"declared C_h3={it['C_h3']} outside [-1,1] (self-report)")
        for k in ("P_h2", "P_h3"):
            if not 0.05 <= float(it[k]) <= 0.95:
                fail("norm-range-w", f"declared {k}={it[k]} outside [.05,.95] (self-report)")
    except (KeyError, TypeError, ValueError):
        fail("norms-missing-w", "norm field missing or non-numeric (self-report)")

    # minimal-pair: RQ2b minimal pair (audit C-2)
    ov, fil = (it.get("ov_word") or "").strip(), (it.get("fil_word") or "").strip()
    if ov or fil:
        if ov and not overlap({stem(ov.lower())}, content(h3, names)):
            fail("minimal-pair", f"ov_word {ov!r} is not a content word of h3")
        if fil and overlap({stem(fil.lower())}, content(h2, names) | content(h3, names)):
            fail("minimal-pair", f"fil_word {fil!r} overlaps with h2/h3")
        if ov and fil and abs(syllables(ov) - syllables(fil)) > 1:
            fail("minimal-pair-w", f"syllable count {syllables(ov)} vs {syllables(fil)} (>1 apart, warning)")

    return bad


H3_CLASS = {"can": "ability", "could": "ability",
            "is": "property", "are": "property", "was": "property",
            "has": "possession", "have": "possession",
            "owns": "possession", "own": "possession",
            "holds": "possession", "keeps": "possession"}


def h3_class(it: dict) -> str:
    """Predicate type of h3: the declared value if present, else guessed from the first word."""
    if it.get("h3_class"):
        return it["h3_class"]
    w = words(it.get("h3", ""))
    if not w:
        return "?"
    first = w[0].lower()
    if first in H3_CLASS:
        return H3_CLASS[first]
    return "event" if first.endswith("ed") else "activity"


def check_balance(items: list[dict]) -> list[str]:
    """Set level: is the h3 predicate type skewed to one class (audit D-1)?

    If every relevance conjunct has the same predicate type, relevance and that type are confounded:
    a coarser confound than lexical overlap, invisible in any single item.
    """
    if len(items) < 4:
        return []
    c = Counter(h3_class(it) for it in items)
    top, n = c.most_common(1)[0]
    msgs = [f"h3 predicate-type distribution: {dict(c)}"]
    if n > len(items) / 2:
        msgs.append(f"**class-balance violation**: {top!r} is {n}/{len(items)}, over half. "
                    f"relevance is confounded with predicate type")
    return msgs


def check_tokenizers(items: list[dict]) -> list[str]:
    """Re-check length-match with real tokenizers; skipped without transformers."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return ["(transformers not installed; tokenizer length-match check skipped)"]
    msgs = []
    for name in TOKENIZERS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # network or permissions
            msgs.append(f"(failed to load tokenizer {name}: {type(exc).__name__})")
            continue
        for it in items:
            n2 = len(tok.encode(" " + it.get("h2", ""), add_special_tokens=False))
            n3 = len(tok.encode(" " + it.get("h3", ""), add_special_tokens=False))
            if abs(n2 - n3) > 1:
                msgs.append(f"length-match[{name.split('/')[-1]}] {it['scenario_id']}: {n2} vs {n3}")
    return msgs


# ------------------------------------------------------------------ main
def load_existing() -> list[dict]:
    """Load the nine items of sim/scenarios9.py in this schema."""
    sys.path.insert(0, str(PROJ / "sim"))
    from scenarios9 import SCEN, HUMAN_NORMS  # type: ignore

    out = []
    for sid, d in SCEN.items():
        nm = HUMAN_NORMS.get(sid, {})
        # the source has no domain field; left empty so domain-cap does not misfire
        org = chung_frame.extract_org(d["ctx"])
        out.append(dict(scenario_id=sid, domain="", paradigm=d["paradigm"], org=org or "",
                        tgt=d["tgt"], e=d.get("e") or "", h1=d["h1"], h2=d["h2"], h3=d["h3"],
                        Q="", ctx=d["ctx"], ctx_irr="",
                        P_h2=nm.get("P_h2"), P_h3=nm.get("P_h3"),
                        C_h2=nm.get("C_h2"), C_h3=nm.get("C_h3")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--existing", action="store_true", help="check the nine items of sim/scenarios9.py")
    ap.add_argument("--tokenizers", action="store_true", help="re-check length-match with real tokenizers")
    a = ap.parse_args()

    items = load_existing() if a.existing else [json.loads(p.read_text("utf-8")) for p in a.files]
    if not items:
        print("no items to check", file=sys.stderr)
        return 2

    corpus = {k: Counter() for k in ("pairs", "e", "h1", "h2", "h3", "domain")}
    for it in items:
        corpus["pairs"][(it.get("h2", "").strip().lower(), it.get("h3", "").strip().lower())] += 1
        for f in ("e", "h1", "h2", "h3"):
            corpus[f][(it.get(f) or "").strip().lower()] += 1
        corpus["domain"][(it.get("domain") or "").strip().lower()] += 1

    npass = 0
    rules, noted = Counter(), Counter()
    for it in items:
        allbad = check_item(it, corpus)
        ex = out_of_scope(it)
        exempt = [(r, m) for r, m in allbad if r in ex]
        rest = [(r, m) for r, m in allbad if r not in ex]
        bad = [(r, m) for r, m in rest if not r.endswith("-w")]
        warn = [(r, m) for r, m in rest if r.endswith("-w")]
        sid = it.get("scenario_id", "?")
        if not bad:
            npass += 1
            print(f"PASS  {sid}" + (f"   ({len(warn)} warnings)" if warn else "")
                  + (f"   ({len(exempt)} reference)" if exempt else ""))
        else:
            print(f"FAIL  {sid}" + (f"   ({len(exempt)} reference)" if exempt else ""))
        for rule, msg in bad + warn:
            rules[rule] += 1
            print(f"        {rule}  {msg}")
        for rule, msg in exempt:
            noted[rule] += 1
            print(f"        [out of scope, reference] {rule}  {msg}")

    print("\nSet-level checks:")
    for m in check_balance(items):
        print(f"  {m}")

    if a.tokenizers:
        print("\nlength-match with real tokenizers:")
        for m in check_tokenizers(items):
            print(f"  {m}")

    print(f"\n{npass}/{len(items)} passed")
    if rules:
        print("violated rules: " + ", ".join(f"{k}×{v}" for k, v in rules.most_common()))
    if noted:
        print("out-of-scope GEN_ONLY findings (reference): "
              + ", ".join(f"{k}×{v}" for k, v in noted.most_common()))
        print("  not edited, being replication targets; GEN_ONLY constrains newly generated items only.")
    return 0 if npass == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
