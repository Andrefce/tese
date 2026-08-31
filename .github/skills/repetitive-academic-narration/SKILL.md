---
name: repetitive-academic-narration
description: 'Detect and revise repetitive academic narration in thesis chapters. Use when prose overuses signposting, repeated conclusions, demonstratives such as "This distinction", causal transitions such as "Therefore", or recurring sentence frames, especially in literature review, methodology, and conclusions. Preserves technical terminology, equations, citations, numerical results, and methodological decisions.'
argument-hint: 'Specify chapters, sections, pages, or recurring phrases to review'
user-invocable: true
disable-model-invocation: false
---

# Repetitive Academic Narration

Diagnose repetitive academic narration, obtain the author's approval for a focused candidate set, and then revise selected thesis prose without changing its technical meaning. Target rhetorical repetition rather than unavoidable domain terminology.

## Inputs

Use the user's named files, sections, pages, or phrases as the review boundary. If no boundary is given, inspect `chapters/02-literature-review.tex`, `chapters/03-methodology.tex`, and `chapters/05-conclusions.tex` first. Expand to other chapters only when the same issue is clearly present.

Treat page numbers as approximate because pagination can change after edits. Locate passages through headings, distinctive phrases, or the compiled PDF when necessary.

## Procedure

1. Read the workspace thesis instructions and the full local context around each candidate paragraph.
2. Search within the requested scope for recurring rhetorical frames, including:
   - `This distinction`, `This makes`, `This is important`, and similar demonstrative openings.
   - `Therefore`, `The reason`, `Together`, and repeated causal or summarising transitions.
   - Sentences that announce what a section will do when the following technical content already makes that purpose clear.
   - Conclusions repeated across nearby sections, especially sparse observations, missing through-plane geometry, the SSM prior, the RBF surrogate, and positive wall separation.
3. Separate rhetorical repetition from necessary terminology. Do not replace precise recurring terms merely to create lexical variety. Terms such as `endocardial and epicardial`, `the SSM fit`, `the Laplace field`, `on the reconstructed geometry`, and dataset names may need to remain unchanged.
4. Rank candidate paragraphs by value. Prioritise passages where removing or combining narration improves the argument without losing a premise, qualification, or cross-reference. Do not rewrite the entire chapter by default.
5. Before editing, present a concise diagnosis containing each candidate's file and location, a short quoted opening or identifying phrase, the repetition problem, and the proposed revision operation. Group overlapping candidates and distinguish necessary terminology from rhetorical repetition.
6. Ask the author to approve, reject, or adjust the candidate set. Do not modify thesis prose until approval is given. Treat the suggested 20--30 paragraphs and page list as prioritisation guidance rather than a quota unless the author explicitly requests a fixed target.
7. For each approved candidate, state a local revision goal before editing: remove redundant signposting, merge a repeated conclusion, make the claim direct, or vary an overused sentence frame.
8. Revise the smallest complete unit, usually one paragraph and occasionally two adjacent paragraphs. Prefer these operations:
   - Delete a metadiscursive sentence when the technical statement stands on its own.
   - Move the substantive fact into the subject position instead of opening with `This`.
   - Combine a repeated interpretation with the evidence that supports it.
   - Replace an announced conclusion with the conclusion itself.
   - Vary paragraph rhythm only where it improves logical flow.
9. Re-read the paragraph with its neighbours. Restore any missing logical connection explicitly, but avoid replacing one stock transition with another.
10. Continue only while each edit has a clear rhetorical benefit. Stop when remaining repetition consists mainly of required terminology or concepts that genuinely need restatement in a new context.

## Decision Rules

Keep a repeated statement when it introduces a concept for a different audience, is needed for chapter-level independence, carries a new qualification, or supports a distinct methodological decision.

Condense or remove it when the same claim appears nearby with no new evidence, implication, scope, or limitation.

Keep signposting when it resolves a real structural ambiguity, introduces a non-obvious contrast, or guides the reader across a substantial change in analytical level.

Remove signposting when it merely says that a distinction is important, previews the next sentence, or repeats the section heading in prose.

If a revision could alter a technical claim, numerical result, equation, citation meaning, experimental condition, model version, or limitation, preserve the original wording and flag the passage for author review instead.

## Thesis Constraints

- Write in clear, formal academic English with limited jargon.
- Use first-person plural for the author's contributions and third person for prior work.
- Preserve LaTeX commands, labels, citations, equations, figure references, and table references.
- Do not introduce or remove citations unless the user explicitly requests source work.
- Do not change equations, hyperparameters, numerical results, methodological decisions, or limitations as part of stylistic revision.
- Call the reconstruction system `the model`, `the proposed model`, or `the proposed approach`; never give it a proper name.
- Treat the existing training-history and epoch material as fixed v2 evidence.
- Do not turn prose into bullet-point lists.
- Keep each section or subsection opening compliant with the repository's `\noindent` convention.

## Quality Checks

Before approval, verify that every proposed edit identifies a specific rhetorical problem and that no candidate is included merely because it repeats required terminology.

After editing, check the revised scope for:

1. Lower frequency of the targeted rhetorical openings and transitions.
2. No consecutive paragraphs using the same argumentative frame.
3. No loss of technical premises, qualifications, limitations, or causal links.
4. No artificial synonym substitution for established domain terms.
5. Unchanged citations, labels, equations, numerical values, and model-version claims unless separately requested.
6. Natural variation in sentence and paragraph structure without informal language.

Review the diff specifically for accidental LaTeX changes. For non-trivial edits, run `latexmk main.tex`, then inspect `main.log` for undefined citations, undefined references, and new overfull `\hbox` warnings. Report the edited scope, the main kinds of repetition reduced, and any passages left unchanged because their repetition was technically necessary.