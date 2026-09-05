"""The Chung et al. (2024) context carrier: the string every new item must inherit.

RQ1 stimuli must be **based on** the Chung paradigm; they are not free writing.
The four contexts in Chung's Appendix B/C **differ in four slots only and are byte-identical otherwise.**
This file is the single source of that fact, and `verify()` re-checks it on every run.

    python3 sim/chung_frame.py        # verify the carrier and print the slot table

Why the carrier is fixed
  1. Comparability with Chung's human data. A different carrier is no longer "the same manipulation".
  2. Noise control. Context length, syntax and speaker role become constants across items and leave the between-item variance.
  3. Fewer degrees of freedom. The generator can touch only the ORG slot, so any
     lexical-overlap (no-overlap) violation can arise in one place only.
"""

# byte-identical to Chung's original except for the four slots.
CARRIER = ("{CONSULTANT} is a consultant doing research for {ORG}, trying to discover new "
           "target groups in Europe for the company to market to. She calls a randomly "
           "selected person, {TGT}, and starts asking {TGT} questions. She finds out that {E}")

# the A-B paradigm (Tentori Exp3) has no evidence e. Chung used M-A only, so this variant
# **is ours**: the minimal departure that drops the last clause of the carrier. The paper says so.
CARRIER_AB = ("{CONSULTANT} is a consultant doing research for {ORG}, trying to discover new "
              "target groups in Europe for the company to market to. She calls a randomly "
              "selected person, {TGT}, and starts asking {TGT} questions.")

# slot values of Chung's four originals (read from Appendix B/C, cross-checked against sim/scenarios9.py)
CHUNG_SLOTS = {
    "violin":            dict(CONSULTANT="Adina",   ORG="an umbrella company"),
    "swiss-man":         dict(CONSULTANT="Natalie", ORG="a car company"),
    "italian-undergrad": dict(CONSULTANT="Maria",   ORG="an American travel agency"),
    "swedish-woman":     dict(CONSULTANT="Becky",   ORG="a shampoo company"),
}

# fixed string spans of the carrier with the slots removed; the carrier-intact check in auto_checks.py uses them.
FIXED_SPANS = (
    " is a consultant doing research for ",
    ", trying to discover new target groups in Europe for the company to market to. "
    "She calls a randomly selected person, ",
    ", and starts asking ",
    " questions. She finds out that ",
)
# A-B lacks the final clause
FIXED_SPANS_AB = FIXED_SPANS[:3] + (" questions.",)


def build(consultant: str, org: str, tgt: str, e: str | None = None) -> str:
    """Fill the slots of the carrier to make the context paragraph. Every new item's ctx must pass through here.

    If e is None the A-B variant (CARRIER_AB) is used.
    """
    t = CARRIER if e else CARRIER_AB
    out = (t.replace("{CONSULTANT}", consultant).replace("{ORG}", org)
           .replace("{TGT}", tgt))
    return out.replace("{E}", e) if e else out


def conforms(ctx: str, paradigm: str = "MA") -> list[str]:
    """List of carrier violations; empty means pass."""
    spans = FIXED_SPANS_AB if paradigm == "AB" else FIXED_SPANS
    return [f"fixed span missing: {sp[:48]!r}..." for sp in spans if sp not in ctx]


def extract_org(ctx: str) -> str | None:
    """Recover the ORG slot from a carrier-conforming context."""
    a, b = FIXED_SPANS[0], FIXED_SPANS[1]
    if a not in ctx or b not in ctx:
        return None
    return ctx.split(a, 1)[1].split(b, 1)[0]


def verify() -> bool:
    """Check that Chung's four originals are reconstructed exactly by this carrier."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scenarios9 import SCEN

    ok = True
    print(f"{'scenario':20s} {'rebuilt':>7s}  ORG slot")
    for sid, slots in CHUNG_SLOTS.items():
        d = SCEN[sid]
        got = build(slots["CONSULTANT"], slots["ORG"], d["tgt"], d["e"])
        same = got.strip() == d["ctx"].strip()
        ok &= same
        print(f"{sid:20s} {'PASS' if same else 'FAIL':>6s}  {extract_org(d['ctx'])!r}")
    return ok


if __name__ == "__main__":
    import sys
    print("Chung context carrier verification\n")
    ok = verify()
    print(f"\ncarrier: {CARRIER}\n")
    print("result:", "PASS: all four rebuilt from the four slots alone" if ok else "FAIL")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------- QUD conversion
# proposal §2.12.2: turn the verb phrase of h3 into a polar question.
#   1. split finiteness   VP -> AUX + PRED  (do-support if needed)
#   2. subject-aux inversion  AUX S PRED
#   3. complement closure     "..., or AUX S not?"
# R = 0.5 is a theorem about this string form, not an empirical claim.
#
# only copulas, auxiliaries and have can be handled safely by machine.
# lexical verbs need lemmatisation, which the string does not determine.
# unhandled cases return None and are written by hand (§2.12.5).
import re as _re

_PRON = {"he": ("he", "him"), "she": ("she", "her"), "they": ("they", "them")}


# they takes plural agreement; needed for targets of unknown gender (V. the surgeon).
_AGREE = {"they": {"is": "are", "was": "were", "does": "do", "has": "have"}}


def _agree(aux: str, pron: str) -> str:
    return _AGREE.get(pron, {}).get(aux.lower(), aux)


def make_qud(tgt: str, vp: str, pronoun: str = "they") -> str | None:
    """Build a polar QUD from verb phrase vp; None if it cannot be done automatically.

    Negated vps are rejected; in a complement design the QUD must be built from the positive side (h2).
    """
    vp = (vp or "").strip().rstrip(".")
    p = _PRON.get(pronoun, ("they", "them"))[0]
    if not vp or _re.search(r"\bnot\b", vp):
        return None
    first, _, rest = vp.partition(" ")
    low = first.lower()

    # the main clause agrees with the **proper name** (singular), the tag with the **pronoun**.
    # with singular they, "Does Robin ..., or do they not?" is correct.
    if low in {"is", "are", "was", "were"}:
        main = {"are": "is", "were": "was"}.get(low, low)
        return f"{main.capitalize()} {tgt} {rest}, or {_agree(main, p)} {p} not?"
    if low in {"can", "could", "will", "would"}:
        return f"{first.capitalize()} {tgt} {rest}, or {low} {p} not?"
    if low in {"has", "have"}:
        # possessive has -> do-support; perfect has + participle is excluded
        if _re.match(r"(been|had|gone|done|\w+ed)\b", rest):
            return None
        return f"Does {tgt} have {rest}, or {_agree('does', p)} {p} not?"
    return None                      # lexical verb: needs lemmatisation, written by hand


def is_complement_pair(h2: str, h3: str) -> bool:
    """Is h3 the negation of h2 (Tentori Exp1 form)?

    **This form cannot support the answerhood contrast.** h2 and h3 are two cells of the same QUD, so
    both answer it, and J1b ("h2 does not answer Q") is impossible in principle.
    A context effect can still be measured but cannot be read as "relevance = answerhood".
    """
    strip = lambda t: _re.sub(r"\bnot\s+", "", (t or "")).strip().lower()
    return bool(h2) and bool(h3) and strip(h2) == strip(h3) and h2.strip() != h3.strip()
