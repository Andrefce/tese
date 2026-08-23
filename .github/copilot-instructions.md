# Project Guidelines — ISTA-IUL Master's Thesis

This workspace is a LaTeX Master's thesis at **Iscte – Instituto Universitário
de Lisboa, Escola de Tecnologias e Arquitetura (ISTA-IUL)**.

**Topic:** 3D reconstruction of the left ventricle from 2D short-axis (SAX)
cardiac MRI slices, and measurement of myocardial wall thickness (SSM / GNN
approaches). Experiments live in Jupyter notebooks alongside the LaTeX source.

When the task is to draft, expand, edit, or review thesis content (anything
under [chapters/](chapters), [frontmatter/](frontmatter), or
[appendices/](appendices)), load the **`ista-tese-mestrado`** skill and follow
it. For sourcing or summarising papers, delegate to the
**Thesis Researcher** subagent.

## Build

- Compile with `latexmk main.tex` (XeLaTeX config in [.latexmkrc](.latexmkrc)).
- Pipeline: XeLaTeX → biber → XeLaTeX → XeLaTeX.
- After non-trivial edits, rebuild and read `main.log` for `Citation undefined`,
  `Reference … undefined`, and overfull `\hbox` warnings before reporting done.

## Source Layout

- [main.tex](main.tex) — preamble + document skeleton. Edit only for global
  package or numbering changes.
- [frontmatter/](frontmatter) — cover, dedication, acknowledgments, **resumo
  (PT)**, abstract (EN).
- [chapters/](chapters) — `01-introduction` … `05-conclusions`. Add new
  chapters by `\input`-ing them from `main.tex`.
- [appendices/](appendices) — supplementary material, after `\appendix`.
- [bibliography/references.bib](bibliography/references.bib) — single biblatex
  database. Always add the `.bib` entry **before** the `\cite{}`.
- [images/](images) — figures referenced via `\graphicspath{{images/}}`.
- [drafts/](drafts) — **scratch only**. Never `\input` from `main.tex`; never
  treat its content as authoritative — the writer may read it as raw material.
- [templates/](templates) — read-only reference of the official ISCTE template.

## Language Policy

- Body language is **English**.
- The reconstruction model has **no proper name**. In prose, captions, tables,
  and figure labels, call it ``the model'', ``the proposed model'', or ``the
  proposed approach''. Never call it ``CardioSDF''; lowercase occurrences in
  legacy filenames, checkpoint paths, and Python identifiers are implementation
  details only.
- Use only the v2 model for thesis experiments and Results. Never compare v2
  against the earlier model. Earlier checkpoints and outputs are implementation
  history only and must not appear as thesis evidence.
- Do not alter or re-audit the existing epoch counts, training-history metrics,
  or training-curve figure; treat that material as v2. Current revision work
  adds new Results content only.
- Use mesh renders when a result needs spatial geometry. Use Matplotlib for new
  quantitative metric plots.
- [frontmatter/resumo.tex](frontmatter/resumo.tex) is **mandatory in
  Portuguese** at ISCTE — never translate, remove, or change its
  `\selectlanguage{portuguese}` switch.

## LaTeX Conventions

- Citations via `biblatex` + `biber`, **APA** style (set in [main.tex](main.tex)).
  Use `\textcite{key}` for narrative citations, `\parencite{key}` for
  parenthetical, and `\cite{key}` only as a last resort.
- Cross-references with `\cref{}` / `\Cref{}` (cleveref). Every chapter,
  section, figure, table, equation, and listing gets a `\label{}` using the
  prefixes `ch:`, `sec:`, `subsec:`, `fig:`, `tab:`, `eq:`, `lst:`, `app:`.
- Each section/subsection starts with `\noindent` (template requirement).
- Figures: `\caption` **below** the graphic, `\label` after the caption.
- Tables: use `booktabs` (`\toprule` / `\midrule` / `\bottomrule`); caption
  **above** the table.
- Math: prefer `equation` (or `align`) with a `\label{eq:…}`; refer to it via
  `\cref{eq:…}`, never a raw `(\ref{…})`.
- Code blocks via the configured `lstlisting` `thesisCode` style (see
  [main.tex](main.tex)); always set `caption` and `label`.
- **No bullet-point lists** (`itemize`, `enumerate`) in thesis prose. Express
  ideas in flowing paragraphs. Lists are acceptable only in the methodology
  review section for search-term groupings if strictly necessary.

## Bibliography Workflow

- New citation → first append a complete entry to
  [bibliography/references.bib](bibliography/references.bib), preferring DOI;
  then cite. **Never invent or guess entries.** If a source cannot be
  verified, delegate to the Thesis Researcher subagent or stop and ask.

## Writing Style

- Formal academic English; first-person plural (“we”) for the author's
  contributions, third person for prior work.
- Each chapter starts with a 1–2 sentence overview paragraph (already present
  as a placeholder in every chapter file).
- No emojis, no informal contractions, no marketing language in `.tex` files.

## Anti-patterns

- Editing the official cover layout in [frontmatter/cover.tex](frontmatter/cover.tex)
  beyond the `% TODO` placeholders.
- Removing or translating [frontmatter/resumo.tex](frontmatter/resumo.tex).
- `\input`-ing anything from [drafts/](drafts) into [main.tex](main.tex).
- Committing build artifacts (`*.aux`, `*.log`, `*.bbl`, `main.pdf`) — they
  are gitignored.
- Fabricated citations or paraphrases without a `.bib` entry.
