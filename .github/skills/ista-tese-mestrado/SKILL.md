---
name: ista-tese-mestrado
description: "Use when drafting, expanding, editing, reviewing, or restructuring chapters of the ISTA-IUL (Iscte – Instituto Universitário de Lisboa) Master's thesis written in LaTeX. Covers ISCTE structural rules (cover, mandatory Portuguese resumo, frontmatter order), biblatex APA citations, figure/table/equation conventions, and converting Jupyter notebook experiments into Methodology and Results sections. Topic of this thesis: 3D reconstruction of the left ventricle from 2D SAX cardiac MRI slices and myocardial wall-thickness measurement (SSM / GNN)."
argument-hint: "Describe the chapter, section, or material to write or revise"
---

# ISTA-IUL Master's Thesis (LaTeX)

Authoritative procedure for writing this thesis. Defer to the rules here over
generic LaTeX habits.

## When to Use

- Drafting or revising any file under [chapters/](../../../chapters),
  [frontmatter/](../../../frontmatter), or [appendices/](../../../appendices).
- Adding or fixing citations, figures, tables, equations, code listings, or
  cross-references.
- Turning a Jupyter notebook (cells + outputs) into thesis prose for
  Methodology or Results.
- Reviewing structure or ISCTE compliance.

## When NOT to Use

- Editing the official cover layout in
  [frontmatter/cover.tex](../../../frontmatter/cover.tex) — only fill the
  `% TODO` placeholders.
- Importing scratch material from [drafts/](../../../drafts) into the build.
- Modifying [templates/](../../../templates) — read-only reference.

## Document Skeleton (authoritative order)

Mirror what is already in [main.tex](../../../main.tex):

1. `\frontmatter` — cover, dedication, acknowledgments, **resumo (PT)**,
   abstract (EN), `\tableofcontents`, `\listoffigures`, `\listoftables`.
2. `\mainmatter` — chapters 1–5 (Introduction, Literature Review, Methodology,
   Results & Evaluation, Conclusions).
3. `\backmatter` — `\printbibliography`, then `\appendix` + appendices.

Add a new chapter by creating `chapters/NN-name.tex` and `\input`-ing it from
`main.tex`. Never `\input` from `drafts/`.

## Per-chapter Checklists

See [chapter-structure.md](./references/chapter-structure.md) for the required
sections and length heuristics for each chapter, distilled from the ISCTE
template TODO scaffolding.

## Frontmatter Rules

See [frontmatter-rules.md](./references/frontmatter-rules.md). Key invariants:

- **`resumo.tex` is in Portuguese and stays in Portuguese.** Never translate
  it, never remove its `\selectlanguage{portuguese}` … `\selectlanguage{english}`
  pair.
- Cover follows the ISCTE three-page layout (main logo → department logo →
  date). Only edit the `% TODO` placeholders.
- Abstract and Resumo each end with a `\textbf{Keywords:}` /
  `\textbf{Palavras-chave:}` line of 3–5 terms.

## LaTeX Conventions

See [figures-tables-math.md](./references/figures-tables-math.md) for
copy-paste templates. Hard rules:

- Each section/subsection body starts with `\noindent` (ISCTE template).
- One blank line between paragraphs; never `\\` to break paragraphs.
- Cross-references via `\cref{}` / `\Cref{}` from `cleveref`. Never write a
  bare `\ref{}` inside parentheses.
- Label prefixes: `ch:`, `sec:`, `subsec:`, `fig:`, `tab:`, `eq:`, `lst:`,
  `app:`. Every floating element gets a label.
- Figures: `\caption` **below** the graphic, then `\label`. Place files in
  [images/](../../../images) (already on `\graphicspath`).
- Tables: `booktabs` (`\toprule` / `\midrule` / `\bottomrule`), no vertical
  rules. Caption **above** the table.
- Math: use `equation` (single line) or `align` (multi-line) with a
  `\label{eq:…}`. Reference with `\cref{eq:…}`.
- Code: use the preconfigured `thesisCode` `lstlisting` style; always set
  `caption=` and `label=`.

## Citations (biblatex + biber, APA)

See [citations-apa.md](./references/citations-apa.md). One workflow:

1. **Add the bib entry first.** Append a complete record to
   [bibliography/references.bib](../../../bibliography/references.bib),
   preferring `doi` when available. Templates live in
   [assets/bib-entries.bib](./assets/bib-entries.bib).
2. **Cite.** Use `\textcite{key}` for narrative citations
   (`\textcite{author2024}` shows ‘Author (2024)’) and `\parencite{key}` for
   parenthetical (`(Author, 2024)`). `\cite{key}` is a fallback.
3. **Verify.** Rebuild and check `main.log` for `Citation … undefined` and
   `package biblatex Warning: Empty bibliography`.
4. **Never invent entries.** If a source cannot be verified, delegate to the
   `thesis-researcher` subagent or stop and ask.

## Notebook → Thesis Workflow

See [notebook-to-thesis.md](./references/notebook-to-thesis.md). Summary:

- Read the notebook with the available notebook tools (do **not** execute
  cells unless the user asks).
- For Methodology: extract the *intent* of each step (preprocessing,
  segmentation, mesh fitting, GNN training) into prose, not code dumps. A
  short `lstlisting` snippet is fine for a key algorithm.
- For Results: pull metrics from the notebook outputs (e.g. Dice, Hausdorff,
  wall-thickness MAE in mm) into a `booktabs` table. Reference each figure
  the notebook produced; do **not** auto-copy images into
  [images/](../../../images) — ask the user before moving files.
- Domain vocabulary: LV (left ventricle), endocardium, epicardium, SAX (short
  axis), SSM (statistical shape model), GNN (graph neural network), AHA-17.
  Define each acronym at first use: `Statistical Shape Model (SSM)`.

## Build & Validate

```bash
latexmk main.tex
```

After non-trivial edits:

1. Rebuild and capture `main.log`.
2. Search for `Citation \w+ undefined`, `Reference \w+ undefined`,
   `Empty bibliography`, `Overfull \\hbox`.
3. If new bib entries were added, latexmk runs biber automatically; if it
   doesn't update, run `latexmk -C` and rebuild.

## Style Discipline

- Formal academic English; first-person plural (“we”) for our contributions,
  third person for prior work.
- Each chapter opens with a 1–2 sentence overview paragraph (the placeholder
  already present in every chapter file).
- No emojis, no informal contractions, no marketing language.
- One short comment per non-obvious LaTeX block; never narrate the obvious.

## Hard Don'ts

- Fabricating citations or paraphrasing a source without a verified `.bib`
  entry.
- Editing the cover layout outside the `% TODO` placeholders.
- Translating, removing, or re-tagging the language of `resumo.tex`.
- `\input`-ing anything from [drafts/](../../../drafts) into `main.tex`.
- Committing `*.aux`, `*.log`, `*.bbl`, `*.bcf`, `*.run.xml`, or `main.pdf`
  (they are gitignored — keep it that way).

## Snippets & References

- [chapter-template.tex](./assets/chapter-template.tex) — empty chapter.
- [figure-snippet.tex](./assets/figure-snippet.tex) — standard figure block.
- [table-snippet.tex](./assets/table-snippet.tex) — `booktabs` table.
- [bib-entries.bib](./assets/bib-entries.bib) — APA-friendly entry stubs by
  type.
- [frontmatter-rules.md](./references/frontmatter-rules.md)
- [chapter-structure.md](./references/chapter-structure.md)
- [citations-apa.md](./references/citations-apa.md)
- [figures-tables-math.md](./references/figures-tables-math.md)
- [notebook-to-thesis.md](./references/notebook-to-thesis.md)
