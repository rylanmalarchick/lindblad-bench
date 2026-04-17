# Manuscript Voice Guide

Purpose: keep the JCP draft in Rylan's actual writing voice rather than sliding into generic AI-journal prose.

Primary source anchors:

- `/home/rylan/career/writing_style/VOICE_GUIDE.md`
- `/home/rylan/school/archive/old_school_drive/highschool/dcush/whattheyfoughtfor.txt`
- `/home/rylan/school/archive/spring2024/COM221/MalarchickCOM221ReportFinal.pdf`

## Core register

- Direct, educated, and a little unpolished.
- State the point early.
- Do not sound like grant prose, corporate prose, or review-article filler.
- Treat the reader as technically literate.

## Sentence and paragraph rules

- Put the thesis sentence first, or very close to first.
- Follow the claim with the concrete evidence immediately after.
- Let paragraphs run long when the argument needs it; do not break them up just to look modern.
- Use short sentences sparingly for emphasis.
- Semicolons are acceptable. Em dashes are not.

## Manuscript-specific style

- The paper should read like an argument, not a feature list.
- Each subsection should answer one real question:
  - what is the computational regime,
  - what hardware actually helps,
  - what fails to help,
  - why the result matters for control software authors.
- When a result is weak or mixed, say so directly.
- Do not hide the `d=27` GRAPE result. Explain it.
- When a methodological caveat matters, state it in plain language instead of burying it in hedging.

## Preferred rhetorical structure

For most technical paragraphs:

1. Claim.
2. Measurement or implementation fact.
3. Interpretation.

For example:

- "The GPU does not win in host-visible time in the measured range. Kernel-only timing does cross the CPU at larger batches, which means the missing speedup is overhead, not arithmetic."

That is the target rhythm.

## Words and transitions to prefer

- "This matters because..."
- "The result is..."
- "That difference shows..."
- "In this regime..."
- "By contrast..."
- "For the current implementation..."
- "The important point is..."

## Things to avoid

- "state-of-the-art"
- "cutting-edge"
- "novel framework"
- "leverages"
- "facilitates"
- "furthermore"
- "moreover"
- "notably"
- "it is worth noting"
- "compelling"
- "transformative"
- rhetorical questions
- fake certainty where the data do not support it

## Claim discipline

- Separate measured fact from inference.
- If a claim depends on released QuTiP rather than unreleased or documented-only features, say that explicitly.
- If a figure uses kernel-only timing, label it as kernel-only timing.
- If a result depends on batch parallelism, do not generalize it to single-trajectory control loops.
- If the FPGA result is about deterministic latency, do not drift into throughput language.

## Section-level guidance

Introduction:

- Start from the actual computational problem, not general quantum hype.
- Establish the gap quickly: small dense Lindblad propagation is not large-BLAS HPC, and the hardware question is unresolved.

Methods:

- Be plain and exact.
- Name the timing model, data path, and fairness choice without excess apology.

Results:

- Lead with the result.
- Then give the number.
- Then say what the number means.

Discussion:

- Synthesize operating regimes.
- Give actionable recommendations for software authors.
- Keep the discussion grounded in the measured hardware, not in imagined future devices.

Conclusion:

- End on the practical lesson, not on abstract ambition.

## Final filter

Before accepting any paragraph, ask:

- Would Rylan actually write this?
- Does the paragraph say something concrete?
- Is the strongest claim traceable to a figure, table, or log already in the repo?
- If a reviewer pressed on the wording, could the sentence be defended from the data?
