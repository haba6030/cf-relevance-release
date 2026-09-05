# P-GEN: item generation prompt

Sections 2–4 of `00_item_spec.md` turned into a prompt. Released in full with the paper.
The generator **must not coincide with the judge models**, nor with the models under test.


> **Revised 2026-08-20.** Constraints are split into **HARD** and **PREFERENCE** tiers.
> `single-clause`, `no-progressive`, `numeric-threshold` and `same-frame` come from the RQ3 stimulus-bank specification and are not gates for RQ1 items;
> the prompt keeps them **for the record only**. Only `no-complement` (complement pairs) stays hard.
> `AUDIT_criteria.md` §G.

Generate **one** item per call. Asking for several at once makes them alike and produces `domain` duplicates
and near-duplicate `(h2,h3)` pairs.

---

## SYSTEM

```
You are constructing experimental stimuli for a psycholinguistics study on the
conjunction fallacy. You are not answering the items; you are writing them.

The specification has two tiers, and it tells you which is which.

  HARD        Violating one voids the measurement. Each HARD constraint carries the
              evidence for it in the text. A machine gate enforces it after you emit,
              so a violation costs a regeneration, not a judgement call.
  PREFERENCE  Follow it unless it fights a HARD constraint or wrecks the sentence.
              Nothing enforces these. If you break one, say so in `notes`.

Do not treat PREFERENCE items as HARD. Some of them are violated by the published
stimuli this study replicates, which is why they are not gates.

Output exactly one JSON object and nothing else. No prose, no markdown fence,
no explanation outside the JSON.
```

## USER

````
# The item you are writing

A participant sees a preamble and then options.

  no-context :  e                         (the evidence sentence alone)
  context    :  the carrier paragraph      (which ends with e)

Two task forms are run, both from Chung et al. (2024):

  Exp2 form, three-way:   base = h1
                          option2 = h1 + " and " + h2
                          option3 = h1 + " and " + h3
  Exp1 form, two-way:     base = h1  against  h1 + " and " + h_i, one h_i per item

Both forms need all of h1, h2, h3. You must supply all three. The measured quantity
compares each conjunction against `base`, so h1 is not filler and cannot be dropped.

## The one asymmetry that makes an item work

  Without ctx, option2 must be the tempting choice, because e CONFIRMS h2.
  With ctx, option3 must become the tempting choice, because h3 ANSWERS the
  question that ctx puts on the table.

Two different forces, attached to two different conjuncts. An item that lacks this
asymmetry contributes nothing and adds variance. Check it before you check anything else.

## The frozen carrier -- read this before anything else

You do NOT write the context paragraph. Chung et al. (2024) used one paragraph for
every scenario, identical to the byte except for four slots. We inherit it exactly.

  {CONSULTANT} is a consultant doing research for {ORG}, trying to discover new
  target groups in Europe for the company to market to. She calls a randomly
  selected person, {TGT}, and starts asking {TGT} questions. She finds out that {E}

Chung's four fillings of {ORG}:

  an umbrella company / a car company / an American travel agency / a shampoo company

Every word outside the four slots is fixed. Do not paraphrase it, do not add a
sentence, do not split it in two. Keeping it fixed is what makes the new items
comparable to Chung's human data, and it removes paragraph length, syntax and
speaker role from the between-item variance.

So writing a new item comes down to one design problem: WHAT GOES IN {ORG}.

  {ORG} must make h3 worth knowing, and must share no content word with h3.

  Chung filled it by naming the thing h3 names:
      h3 = "owns an umbrella"     ORG = "an umbrella company"      <- shares 'umbrella'
  That is the confound. You fill it by naming the SITUATION in which h3 matters:
      h3 = "owns an umbrella"     ORG = "a firm that surveys who copes with sudden downpours"

  {ORG} is a noun phrase with its own article. It may carry a relative clause. It may
  not be a second sentence.

## Slots

  scenario_id  kebab-case, unique
  domain       topic area, one or two words
  tgt          the target person's given name
  e            evidence sentence, begins with tgt, one finite clause. REQUIRED.
               Never omit it. Without e the no-context condition degenerates to
               "Consider a randomly selected person, X." and carries no information,
               which makes it incomparable with items that do have e.
  h1           reference-class setter: a sentence about tgt that picks out a
               narrow population. NOT a hobby. "Dan is an expert mountaineer."
               is right; "Dan likes coffee." is wrong, because the reference class
               must be small enough that the conjunctions are not near-certain.
  h2           verb phrase only, no subject, no final period. Attaches to h1's subject.
  h3           verb phrase only, no subject, no final period.
  Q            the Question Under Discussion, written out as an explicit polar
               question: "PRED, or not?" Q is NOT shown to participants. It exists
               so that the item can be audited.
  consultant   a female given name for the consultant. Not the target's name.
  org          THE ONLY FREE PART OF THE CONTEXT. See "The frozen carrier" above.
               You do not emit `ctx` -- the pipeline builds it from the carrier.
  org_irr      the same slot filled with an organisation unrelated to h2 and h3.
  h3_class     one of: ability / property / event / possession / role  (see class-spread)
  ov_word      one content word taken from h3
  fil_word     a minimal-pair replacement for ov_word: same part of speech, same
               syllable count, comparable corpus frequency, unrelated meaning
  P_h2, P_h3   your estimate of P(h_i | e), in [0,1]
  C_h2, C_h3   your estimate of how much e confirms h_i, on [-10, +10]
  notes        what you traded away, and why. Not shown to judges.

## HARD constraints -- machine-gated

LENGTH  -- the normalised DV divides by token count, so a length ratio far from 1
        moves the measure on its own                                   [gate length-match]
  token-diff  h2 and h3 must differ by at most 1 whitespace token.
  min-tokens  h2 and h3 must each be at least 3 tokens long.
  token-ratio  The ratio of the longer to the shorter must be at most 1.34. "can ski"
      (2) against "has a driving license" (4) is a ratio of 2.0.

      This rule constrains what WE generate. It is not a verdict on the published
      items being replicated -- those are used as printed, unchanged.

LEXICON  -- the most important block                             [gate no-overlap]
  org-vs-h2  No content word of h2 appears in {ORG}.
  org-vs-h3  No content word of h3 appears in {ORG}.
  content-words  Function words, the target's name, and the consultant's name do not
      count as content words.
  slot-only  The carrier itself is fixed, so {ORG} is the only place YOU can violate
      these. Check it word by word before you emit.
  carrier-words  But the carrier has its own vocabulary, and h2/h3 must avoid it too.
      These words are already in every context, so a conjunct containing one of them is
      primed in the context condition and not in the no-context condition:

        ask  call  company  consultant  discover  europe  group  market  new
        question  research  start  target  try

      "identifies waders by call" fails on `call`, because the carrier says
      "She calls a randomly selected person". Rewrite it as "identifies waders by ear".

  org-vs-h2 and org-vs-h3 are what make this hard, and they are non-negotiable.
  If ctx contains a word from h3, reading ctx lowers the model's surprisal on h3's
  tokens for a reason that has nothing to do with relevance. Taking a difference across
  the two context conditions does NOT cancel it, because the shared word is present
  only in the context condition.

  So you may not connect ctx to h3 through vocabulary. You must connect them
  through  world knowledge.

    WRONG   ORG: "an umbrella company"    h3: "owns an umbrella"
            (the word `umbrella` is shared -- rejected)

    RIGHT   ORG: "a firm that surveys who copes with sudden downpours"
            h3: "owns an umbrella"
            (no shared word; the link is that umbrellas are what one owns in
             order to cope with a downpour)

  Write {ORG} last, and write it against this rule from the start. Retrofitting is harder.

PREDICATE CLASS -- HARD at the set level, and the one most easily missed
                                                                    [warning class-balance]
  class-spread  Across the whole item set, h3 must NOT be dominated by one kind of
      predicate. Spread h3 over these classes and let no class take more than half the
      items:

        ability     "can play the trumpet"
        property    "is allergic to shellfish"
        event       "worked a season in Lisbon"
        possession  "owns a folding table"
        role        "is a blood donor"

      A first attempt at this task produced eight items whose h3 were ALL possession
      predicates (six of them beginning "owns"). Possession is the easiest way to
      satisfy X1-X2, so it is what you will reach for. If every h3 is a possession,
      then "relevance" is confounded with "possession predicate" across the whole set
      and no analysis can separate them. Chung's own four items are mixed:
      ability / possession / event / property.

      h2 and h3 within ONE item must still share a syntactic frame (same-frame). The variety
      is BETWEEN items, not within one.

  class-report  Report the class you chose in the `h3_class` field.

FORM -- PREFERENCE unless marked HARD

  Nothing here is machine-gated except no-complement. These come from the RQ3 stimulus bank
  specification, which governs a different item family, so they are guidance rather
  than  gates. Follow them where they cost nothing. Where one fights a HARD constraint,
  the  HARD constraint wins and you note it.

  single-clause  Prefer single finite clauses. No comma, no "and"/"or", in h2 or h3.
  no-modal  Prefer no modal verbs (may, might, should, must, could, would). Note: "can"
      is fine -- an ability predicate needs it, and class-spread asks for those.
  no-vague-quant  Prefer no vague quantifiers (most, many, few, several, often,
      usually).
  no-progressive  Prefer no progressive aspect ("is repairing").
  no-negation  Prefer no negation in h2 or h3.
  no-complement  HARD -- THE COMPLEMENT TRAP. Never write h3 as the
      negation of h2. "is an interpreter" / "is not an interpreter" is the shape to
      avoid. If h3 = not-h2, the two are the two cells of ONE question, so any question
      h3 answers, h2 answers too, and no context can make h3 relevant without making h2
      relevant. The item then measures nothing the design is about, and the defect
      cannot be repaired by rewriting the context. Two items in the earlier set had it;
      both had to be rewritten from the hypothesis up.
  different-matter  HARD -- h2 and h3 must be about DIFFERENT SUBJECT MATTER, not two
      answers to one question. This is no-complement's weaker sibling and it is judged,
      not machine-checked (J2).
  numeric-threshold  Prefer a numeric threshold under a gradable adjective: "earns more
      than 60,000 euros a year" rather than "earns a high salary".
  same-frame  Prefer the same syntactic frame for h2 and h3 (same verb type, same number
      of arguments). Note the tension with class-spread: spreading h3 across predicate
      classes drags h2 along with it, since h2 must match. When they conflict,
      class-spread wins -- it is a set-level confound and same-frame is a nicety.

LEAKAGE -- HARD                                                    [gate no-name-leak, name-neutral]
  name-neutral  tgt's name must not make h2 or h3 more or less likely. No name that
      implies a nationality, gender, or profession that either conjunct mentions.
  org-silent  {ORG} must say nothing about tgt. The carrier already states e; nothing
      else about the target may appear.
  org-not-evidential  {ORG} must not make h2 or h3 more or less PROBABLE. It changes
      what is being asked, not what is known. If a reader would update their probability
      estimate for h2 or h3 after reading the context, the item is void.

CONTEXT SHAPE -- HARD, constraints on {ORG}, since the rest is frozen
                                                              [gates qud-implicit, org-no-overlap]
  qud-unspoken  Q must not appear verbatim inside {ORG}. The carrier raises Q; it never
      utters it.
  org-shape  {ORG} is 4-14 words, a single noun phrase, no full stop inside it.
  org-no-overlap  {ORG} shares no content word with h2 and no content word with h3.
  org-plausible  {ORG} names a purpose an ordinary organisation could have. Nothing
      evasive, nothing that hints at what the experiment is testing.
  org-no-target  {ORG} says nothing about the target person.

NORMS -- PREFERENCE. These are YOUR estimates, so they cannot gate anything.
        They are recorded and later replaced by measured values (`run_norm`).
  confirms-h2  Aim for C_h2 >= +2.0. e must genuinely confirm h2.
  inert-h3  Aim for C_h3 in [-1, +1]. e should be evidentially inert with respect to h3.
  prob-gap  Prefer P_h3 - P_h2 >= 0.15.
  prob-range  Prefer P_h2 and P_h3 both in [0.05, 0.95].

INDEPENDENCE -- HARD. I1-I3 are judged (J2, J1); two-axes is judged and is the one
              most often missed.
  hyp-independent  h2 and h3 must be logically independent: neither entails nor excludes
      the other.
  h1-independent  h1 must entail neither h2 nor h3.
  answers-qud  h3 must answer Q. h2 must not answer Q.
  two-axes  TWO SEPARATE AXES. The link e->h2 (confirmation) and the link ORG->h3
      (relevance) must run through DIFFERENT subject matter.

        good  e "has a degree in violin performance" -> h2 "gives music lessons"
              ORG "surveys who copes with sudden downpours" -> h3 "owns an umbrella"
              (music and rain are unrelated)

        bad   e "is a Russian woman" -> h2 "is an interpreter"
              ORG "helps New Yorkers read Russian documents" -> h3 "is not an interpreter"
              (both links run through Russian-language work: one axis, not two)

      If the two links share their subject matter, confirmation and relevance are not
      separated by the item, and the whole contrast collapses.

## Worked example (this exact item already exists -- do not reproduce it)

{
  "scenario_id": "violin-rain",
  "domain": "weather-preparedness",
  "tgt": "Dan",
  "e": "Dan has a degree in violin performance.",
  "h1": "Dan is an expert mountaineer.",
  "h2": "gives music lessons",
  "h3": "owns an umbrella",
  "Q": "Does Dan own an umbrella, or does he not?",
  "consultant": "Adina",
  "org": "a firm that surveys who copes with sudden downpours",
  "org_irr": "a firm that surveys who sorts household recycling correctly",
  "ov_word": "umbrella",
  "fil_word": "notebook",
  "P_h2": 0.35,
  "P_h3": 0.67,
  "C_h2": 5.6,
  "C_h3": -0.1,
  "h3_class": "possession",
  "notes": "The link from ORG to h3 runs through world knowledge (coping with a
            downpour) rather than through the word 'umbrella', as org-vs-h3 requires.
            Carrier untouched. Cost: the slot reads more bureaucratic than a
            two-word company name. same-frame satisfied by accident (V+NP against V+NP);
            had it conflicted with the predicate-class budget, class-spread would have won."
}

## Your assignment

domain to use:        {DOMAIN}
already-used (h2,h3): {USED_PAIRS}
already-used domains: {USED_DOMAINS}

Write one new item. Before emitting, walk the HARD constraints one at a time against
your own text: the LENGTH, LEXICON, PREDICATE CLASS, LEAKAGE, CONTEXT SHAPE and
INDEPENDENCE blocks, plus no-complement and different-matter.
If you cannot satisfy org-vs-h2 and org-vs-h3 for the h3 you chose, change h3 --
do not weaken the constraint. Then walk the PREFERENCE list (the FORM and NORMS blocks) and record in
`notes` every one you broke and what forced it.

Emit the JSON object only.
````

---

## Operating notes

- `{DOMAIN}` is set by the caller. Left to the model, it clusters on `technology` and `travel`.
- `{USED_PAIRS}` and `{USED_DOMAINS}` are filled from the items accepted so far (unique-item, domain-cap).
- Temperature about 0.9. Generation needs diversity; filtering is done downstream.
- Output goes to `items/candidates/*.json`; only passing items to `items/accepted/`.
- **ORG-vs-h2 / ORG-vs-h3 overlap is the most common failure.** Run `checks/auto_checks.py` first and do not
  send violating items to the judges; judging costs far more than generating.
- The hard/preference split **corresponds one-to-one** to `GATE`/`GEN_ONLY` in `checks/auto_checks.py`.
  Editing the prompt without the checker desynchronises them; change both together.
- **Requirements for the next round** were recorded in the internal audit memo (confound between predicate type and bridge tightness).
  Whether to turn them into rules was left open.
