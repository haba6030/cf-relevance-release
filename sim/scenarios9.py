"""Nine CF scenarios with relevance contexts.

Sources
  Tentori, Crupi & Russo (2013) JEP:General, Appendix A (Exp1) / B (Exp2) / C (Exp3)
  Chung, Dorst, Mandelkern & Mascarenhas (2024) Appendix B/C: contexts for the four Exp2 items

Context status
  chung : Chung's original (read from the appendix)
  new   : written for this project, no human validation; see the model norming in sim/norm9_*.csv

Rules for new contexts
  1. Keep Chung's frame: a consultant looking for target groups sets the QUD
  2. h3 answers that QUD and h2 does not
  3. No lexical overlap with h3 (two of Chung's four originals overlapped)
"""

# paradigm: MA = evidence e given (Tentori Exp1/2) / AB = no e (Exp3)
SCEN = {
    # ---------------- Tentori Exp2 = Chung's four ----------------
    "violin": dict(
        src="Tentori Exp2", paradigm="MA", ctx_status="chung", tgt="Dan",
        e="Dan has a degree in violin performance.",
        h1="Dan is an expert mountaineer.", h2="gives music lessons", h3="owns an umbrella",
        ctx="Adina is a consultant doing research for an umbrella company, trying to discover new "
            "target groups in Europe for the company to market to. She calls a randomly selected "
            "person, Dan, and starts asking Dan questions. She finds out that "
            "Dan has a degree in violin performance."),
    "swiss-man": dict(
        src="Tentori Exp2", paradigm="MA", ctx_status="chung", tgt="Noah",
        e="Noah is a Swiss man.",
        h1="Noah likes making Italian desserts.", h2="can ski", h3="has a driving license",
        ctx="Natalie is a consultant doing research for a car company, trying to discover new "
            "target groups in Europe for the company to market to. She calls a randomly selected "
            "person, Noah, and starts asking Noah questions. She finds out that Noah is a Swiss man."),
    "italian-undergrad": dict(
        src="Tentori Exp2", paradigm="MA", ctx_status="chung", tgt="Carlo",
        e="Carlo is an Italian undergraduate student.",
        h1="Carlo has red hair.", h2="studied abroad in Barcelona in 2017",
        h3="spent his summer holidays in America in 2017",
        ctx="Maria is a consultant doing research for an American travel agency, trying to discover "
            "new target groups in Europe for the company to market to. She calls a randomly "
            "selected person, Carlo, and starts asking Carlo questions. She finds out that "
            "Carlo is an Italian undergraduate student."),
    "swedish-woman": dict(
        src="Tentori Exp2", paradigm="MA", ctx_status="chung", tgt="Alice",
        e="Alice is a Swedish woman.",
        h1="Alice studies in Italy.", h2="works as a model", h3="has brown hair",
        ctx="Becky is a consultant doing research for a shampoo company, trying to discover new "
            "target groups in Europe for the company to market to. She calls a randomly selected "
            "person, Alice, and starts asking Alice questions. She finds out that "
            "Alice is a Swedish woman."),

    # ---------------- Tentori Exp1 (new contexts) ----------------
    # h3 is of the not-h2 form. Tentori used K./J.; names were added in Chung's manner.
    "russian-woman": dict(
        src="Tentori Exp1", paradigm="MA", ctx_status="new", tgt="Katya",
        e="Katya is a Russian woman.",
        h1="Katya lives in New York.", h2="is an interpreter", h3="is not an interpreter",
        ctx="Sofia is a consultant doing research for a language-services firm. Her task is to "
            "locate New Yorkers who might need help reading official documents written in Russian. "
            "She calls a randomly selected person, Katya, and starts asking Katya questions. "
            "She finds out that Katya is a Russian woman."),
    "american-man": dict(
        src="Tentori Exp1", paradigm="MA", ctx_status="new", tgt="Jack",
        e="Jack is an American man.",
        h1="Jack speaks Italian fluently.", h2="is overweight", h3="is not overweight",
        ctx="Sofia is a consultant doing research for a sportswear brand. Her task is to locate "
            "customers for a new line of running gear. She calls a randomly selected person, Jack, "
            "and starts asking Jack questions. She finds out that Jack is an American man."),

    # ---------------- Tentori Exp3 (new contexts, A-B) ----------------
    # no e; h1 itself is the given category.
    "athlete": dict(
        src="Tentori Exp3", paradigm="AB", ctx_status="new", tgt="Robin",
        e=None,
        h1="Robin is engaged in athletic competitions.",
        h2="is younger than 25 years old", h3="has brown hair",
        ctx="Sofia is a consultant doing research for a cosmetics brand that is developing a new "
            "range of dye shades. Her task is to discover new target groups for the company to "
            "market to. She calls a randomly selected person, Robin, and starts asking Robin "
            "questions."),
    "surgeon": dict(
        src="Tentori Exp3", paradigm="AB", ctx_status="new", tgt="V.",
        # h2 = "is male", so a name would leak the answer; Tentori's initial is kept.
        e=None,
        h1="V. works as a surgeon.", h2="is male", h3="is right-handed",
        ctx="Sofia is a consultant doing research for a company that designs precision hand tools "
            "and must decide which grip to manufacture. Her task is to discover new target groups "
            "for the company to market to. She calls a randomly selected person, V., and starts "
            "asking V. questions."),
    "swiss-person": dict(
        src="Tentori Exp3", paradigm="AB", ctx_status="new", tgt="Milo",
        e=None,
        h1="Milo is Swiss.", h2="can ski", h3="has a driving license",
        ctx="Sofia is a consultant doing research for a car rental chain. Her task is to discover "
            "new target groups in Europe for the company to market to. She calls a randomly "
            "selected person, Milo, and starts asking Milo questions."),
}

# note: swiss-man and swiss-person share h2/h3 (can ski / has a driving license).
# Tentori used them in different paradigms (M-A vs A-B); they must not be counted as independent items.
NON_INDEPENDENT = [("swiss-man", "swiss-person")]

# Tentori human norms (Tables 4-5 = Exp2, 7-8 = Exp3). Exp1 was comparative, no values.
HUMAN_NORMS = {
    "violin":            dict(P_h2=.35, P_h3=.67, C_h2=5.6, C_h3=-0.1),
    "swiss-man":         dict(P_h2=.68, P_h3=.83, C_h2=4.7, C_h3=-0.6),
    "italian-undergrad": dict(P_h2=.16, P_h3=.12, C_h2=3.9, C_h3=-0.4),
    "swedish-woman":     dict(P_h2=.19, P_h3=.25, C_h2=2.6, C_h3=-4.1),
    "athlete":           dict(P_h2=.66, P_h3=.59, C_h2=3.3, C_h3=-0.2),
    "surgeon":           dict(P_h2=.77, P_h3=.78, C_h2=2.4, C_h3=0.2),
    "swiss-person":      dict(P_h2=.76, P_h3=.75, C_h2=3.1, C_h3=-0.3),
}


def no_context(d):
    """Introduction for the no-context condition."""
    return d["e"] if d["paradigm"] == "MA" else f"Consider a randomly selected person, {d['tgt']}."


def options(d):
    """The three options: base = h1, h2 = h1 ∧ h2, h3 = h1 ∧ h3."""
    return dict(base=d["h1"],
                h2=f"{d['h1'].rstrip('.')} and {d['h2']}.",
                h3=f"{d['h1'].rstrip('.')} and {d['h3']}.")
