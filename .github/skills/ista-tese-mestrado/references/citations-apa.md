# Citations — biblatex APA

The thesis uses `biblatex` with `style = apa`, `sorting = nyt`, `backend =
biber`. Configured in [main.tex](../../../../main.tex). Pipeline:
pdflatex → biber → pdflatex → pdflatex (`latexmk -pdf` does this).

## Commands (use in this order of preference)

| Command | Output (APA)         | Use when                                  |
|---------|----------------------|-------------------------------------------|
| `\textcite{key}`  | Author (Year)        | Author is the grammatical subject of the sentence. |
| `\parencite{key}` | (Author, Year)       | Citation is parenthetical evidence.       |
| `\cite{key}`      | depends on style     | Last-resort fallback only.                |
| `\textcite[p.~12]{key}` | Author (Year, p. 12) | Page-specific reference.            |
| `\parencite[see][]{key}` | (see Author, Year)  | Add a prenote.                       |

Examples:

```latex
\textcite{ronneberger2015unet} introduce U-Net for biomedical segmentation.
Recent work on cardiac MRI segmentation~\parencite{bernard2018acdc} shows that
\ldots
```

## Workflow

1. **Find** the canonical source (DOI, arXiv, publisher page).
2. **Add** the entry to
   [bibliography/references.bib](../../../../bibliography/references.bib)
   *before* citing. Use a stable key: `firstauthorYEARkeyword` (lowercase, no
   spaces), e.g. `ronneberger2015unet`, `bai2018automated`.
3. **Cite** in the chapter file with `\textcite{}` / `\parencite{}`.
4. **Build** with `latexmk -pdf main.tex` and check `main.log` for
   `Citation '...' undefined` and `package biblatex Warning`.

## Required Fields by Entry Type

Minimum APA-correct fields. Templates with realistic stubs are in
[../assets/bib-entries.bib](../assets/bib-entries.bib).

### `@article` (journal paper)

`author`, `title`, `journal`, `year`, `volume`, `number`, `pages`, `doi`.

### `@inproceedings` (conference paper)

`author`, `title`, `booktitle`, `year`, `pages`, `publisher`, `doi`.

### `@book`

`author` (or `editor`), `title`, `year`, `publisher`, `address`, `isbn`.

### `@incollection` (book chapter)

`author`, `title`, `booktitle`, `editor`, `year`, `publisher`, `pages`.

### `@misc` (arXiv preprint, technical report, dataset)

`author`, `title`, `year`, `eprint`, `archivePrefix = {arXiv}`, `eprinttype =
{arxiv}`, `eprintclass`, plus `howpublished` or `url` and `urldate` if not on
arXiv.

### `@phdthesis` / `@mastersthesis`

`author`, `title`, `year`, `school`, `address`.

## Author Names

- Use `Last, First Middle` form so APA can format ‘Last, F. M.’ correctly.
- Multiple authors: separate with ` and `:
  `author = {Bai, Wenjia and Sinclair, Matthew and Tarroni, Giacomo}`.
- Up to 99 authors are listed (`maxbibnames = 99`); biblatex truncates the
  in-text citation to APA's “et al.” rules automatically.

## DOIs and URLs

- Always include `doi = {10.xxxx/xxxx}` when one exists.
- `url` is suppressed in this style (`url = false` in `main.tex`); use it only
  for sources without a DOI (datasets, software).
- For arXiv: `eprint = {2401.01234}`, `archivePrefix = {arXiv}`.

## Common Pitfalls

- **Capitalisation.** APA preserves bib-file capitalisation; protect proper
  nouns with braces: `title = {{LV} segmentation with {U-Net}}`.
- **Math in titles.** Wrap math in `$...$` and protect with braces:
  `title = {Estimating ${\beta}$ from sparse slices}`.
- **Duplicate keys.** `latexmk` will warn; rename and rebuild.
- **Missing `urldate`.** When you must cite a website, also set
  `urldate = {2025-09-01}`.
- **Inventing entries.** Do not. Delegate to the `thesis-researcher` subagent
  or stop and ask.
