# Per-chapter Structure

What every chapter must contain, with length heuristics. Mirrors the TODO
scaffolding already in `chapters/`.

## Common Rules

- Open with a 1–2 sentence overview paragraph after `\chapter{...}` and
  `\label{ch:...}` (already present as the placeholder).
- Each `\section`/`\subsection` body starts with `\noindent`.
- All chapters, sections, subsections get labels: `ch:`, `sec:`, `subsec:`.
- Cross-reference other parts with `\cref{}` only.

## Chapter 1 — Introduction (`chapters/01-introduction.tex`)

Length: 4–8 pages. Required sections:

1. `\section{Motivation}` — broader clinical / scientific context. For this
   thesis: cardiac MRI as the gold standard, role of LV wall thickness in
   diagnosing infarction and hypertrophy, why automation matters.
2. `\section{Problem Statement}` — the specific gap (e.g. sparse SAX slices,
   anisotropic resolution, manual segmentation cost).
3. `\section{Research Questions}` — RQ1..RQn as an `enumerate` with
   `label=\textbf{RQ\arabic*.}`.
4. `\section{Objectives}` — itemised, action verbs.
5. `\section{Contributions}` — itemised; one item per concrete deliverable
   (model, dataset preparation pipeline, evaluation protocol, …).
6. `\section{Document Structure}` — one short paragraph mapping each chapter
   with `\Cref{ch:…}`.

## Chapter 2 — Literature Review (`chapters/02-literature-review.tex`)

Length: 10–20 pages. Required sections:

1. `\section{Background}` — fundamentals the reader needs (cardiac anatomy,
   MRI SAX acquisition, SSM theory, GNN basics). Define every acronym on
   first use.
2. `\section{Related Work}` — group approaches into named families as
   subsections, e.g.:
   - `\subsection{Atlas- and SSM-based reconstruction}`
   - `\subsection{Deep-learning segmentation pipelines}`
   - `\subsection{Graph-based mesh learning}`
   - `\subsection{Wall-thickness measurement methods}`
   For each family: representative works (`\textcite{}`), strengths,
   limitations.
3. `\section{Summary and Research Gap}` — one paragraph that states what
   nobody has done yet and how this thesis fills the gap.

Reference [drafts/literatureReviewDraft.tex](../../../../drafts/literatureReviewDraft.tex)
and [drafts/literatureThicknessMethods.tex](../../../../drafts/literatureThicknessMethods.tex)
as raw material — distil, do **not** copy-paste, and re-cite from
`references.bib`.

## Chapter 3 — Methodology (`chapters/03-methodology.tex`)

Length: 8–15 pages. Required sections:

1. `\section{Overview}` — high-level pipeline diagram (TikZ or PNG).
2. `\section{System Architecture}` — component diagram + dataflow.
3. `\section{Components}` — one subsection per component, e.g.
   `Preprocessing`, `Segmentation`, `Surface fitting / SSM`, `GNN refinement`,
   `Thickness extraction`. For each: inputs, outputs, key equations
   (`equation` env with `\label{eq:…}`).
4. `\section{Implementation Details}` — Python stack (PyTorch / MONAI /
   PyTorch Geometric / SimpleITK / nibabel / VTK), reproducibility
   (random seeds, hardware), repository layout.

Use `lstlisting` (`thesisCode` style) for ≤ 25-line algorithm excerpts; longer
listings go to an appendix.

## Chapter 4 — Results and Evaluation (`chapters/04-results.tex`)

Length: 10–20 pages. Required sections:

1. `\section{Experimental Setup}` with subsections
   `\subsection{Datasets}` (e.g. ACDC, M\&Ms, UK Biobank — only those actually
   used), `\subsection{Metrics}` (Dice, IoU, Hausdorff, wall-thickness MAE in
   mm — define each formally with an `equation`), `\subsection{Baselines}`.
2. `\section{Results}` — `booktabs` tables for quantitative numbers; figures
   for qualitative comparisons. Bold the best value per metric.
3. `\section{Discussion}` — interpret, do not just restate the table.
4. `\section{Threats to Validity}` — internal (overfitting, data leakage,
   inter-observer variability), external (single-vendor MRI, paediatric vs
   adult).

## Chapter 5 — Conclusions (`chapters/05-conclusions.tex`)

Length: 2–4 pages. Required sections:

1. `\section{Summary}`.
2. `\section{Revisiting the Research Questions}` — one paragraph per RQ from
   `\cref{sec:research-questions}` with the answer this thesis supports.
3. `\section{Contributions}` — restate, now backed by results.
4. `\section{Limitations}`.
5. `\section{Future Work}` — concrete, not aspirational.

## Appendices (`appendices/`)

- Hyper-parameter tables, full per-subject metrics, longer code listings,
  questionnaires.
- Each appendix is a separate `\chapter{}` after `\appendix`, labelled `app:`.
