# Compilation & Build Pipeline

## Engines

| Engine | When to use |
|--------|-------------|
| `pdflatex` | Default; fast; works with most packages |
| `xelatex` | Native UTF-8, system fonts, complex scripts |
| `lualatex` | Modern engine, scripting in Lua; slower |

Stick with **pdflatex** unless you need custom system fonts (then use `xelatex`).

## Build Order

### With biblatex + biber (recommended)

```
pdflatex main
biber main
pdflatex main
pdflatex main
```

### With BibTeX (legacy)

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

### Without bibliography

Two `pdflatex` runs suffice when cross-references stabilise.

### With glossaries/acronyms

```
pdflatex main
makeglossaries main
pdflatex main
pdflatex main
```

## Latexmk (Recommended)

`latexmk` automates the build — it figures out how many passes are needed.

```bash
latexmk -pdf main.tex            # pdflatex
latexmk -xelatex main.tex        # xelatex
latexmk -lualatex main.tex       # lualatex
latexmk -c                       # clean intermediate files (keep PDF)
latexmk -C                       # full clean (remove PDF too)
```

Create a `.latexmkrc` in the project root for project-wide defaults:

```perl
$pdf_mode = 1;           # pdflatex
$bibtex_use = 2;         # always run bibtex/biber
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 %O %S';
@default_files = ('main.tex');
```

## VS Code Integration

Install the **LaTeX Workshop** extension. Sensible workspace settings:

```jsonc
// .vscode/settings.json
{
  "latex-workshop.latex.recipe.default": "latexmk",
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.view.pdf.viewer": "tab",
  "latex-workshop.latex.clean.method": "glob",
  "editor.formatOnSave": true,
  "[latex]": {
    "editor.wordWrap": "on",
    "editor.rulers": [100]
  }
}
```

## Common Build Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `! LaTeX Error: File 'X.sty' not found` | Missing package | Install via TeX Live Manager / MiKTeX |
| `! Undefined control sequence \X` | Typo or missing package | Check spelling; ensure `\usepackage{...}` |
| `! LaTeX Error: Environment X undefined` | `\newtheorem` missing or typo | Check declarations in preamble |
| `Reference 'X' on page N undefined` | Missing `\label` or compile pass | Add label; recompile twice |
| `[?]` in PDF | Same as above | Recompile after adding/fixing label |
| `Overfull \hbox (Xpt too wide)` | Line doesn't fit | Add `-` or `\hyphenation{}`; rephrase |
| `Underfull \hbox (badness 10000)` | Loose spacing | Usually safe to ignore |
| `! File ended while scanning use of \X` | Missing brace or `\end{}` | Check matching delimiters |
| `Missing $ inserted` | Math symbol used outside math mode | Wrap in `$...$` |
| `! Package biblatex Warning: Please rerun LaTeX` | biber/bibtex not run | Run `biber main` then `pdflatex` |

## Reading the Log

After compiling, inspect `main.log` for:
- `! ` lines — errors (must fix)
- `LaTeX Warning:` — undefined references, multiply-defined labels
- `Overfull` / `Underfull` — typography issues

Use VS Code LaTeX Workshop's problems panel to navigate them.

## Output Verification

Before submission:
- [ ] PDF generates with no errors and no warnings about undefined references
- [ ] All figures appear in the correct place
- [ ] Bibliography is complete and properly formatted
- [ ] TOC is up to date (no `??` page numbers)
- [ ] Page numbers transition correctly (Roman → Arabic)
- [ ] Cover has no page number
- [ ] PDF metadata is set (`\hypersetup{pdftitle=..., pdfauthor=...}`)

## Reproducible Builds

Pin the TeX Live / MiKTeX version:
- Document the version used in a `README.md`
- For collaborative work, consider Overleaf (which fixes the TeX Live year)

## Useful Compile Flags

```bash
pdflatex -interaction=nonstopmode main.tex    # don't pause on errors
pdflatex -synctex=1 main.tex                   # forward/inverse search
pdflatex -shell-escape main.tex                # required for minted
pdflatex -file-line-error main.tex             # better error locations
```

## Project Cleanup Before Submission

Remove auxiliary files but keep `.bib`, `.tex`, images, and the final PDF:

```bash
latexmk -C
```

Or via PowerShell:

```powershell
Get-ChildItem -Include *.aux,*.log,*.out,*.toc,*.lof,*.lot,*.bbl,*.blg,*.bcf,*.run.xml,*.synctex.gz -Recurse | Remove-Item
```
