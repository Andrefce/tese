# Citations & References

## Recommended Approach: `biblatex` + Biber

`biblatex` is the modern replacement for the classic `\bibliography{}` + BibTeX flow. It's more powerful, easier to configure, and the recommended choice for new theses.

### Preamble

```latex
\usepackage[
  backend  = biber,
  style    = numeric,        % or: alphabetic, authoryear, ieee, apa, acm
  sorting  = nyt,            % name-year-title
  maxbibnames = 99,
  giveninits  = true,
  natbib   = true,           % enables \citet, \citep
  url      = false,
  doi      = true,
  isbn     = false,
]{biblatex}

\addbibresource{bibliography/references.bib}
```

### In the document

```latex
% Cite a single source
As shown by \textcite{smith2020}, ...
% Output: As shown by Smith (2020), ...

% Parenthetical cite
Recent work has explored this area \parencite{smith2020, jones2021}.
% Output: Recent work has explored this area (Smith, 2020; Jones, 2021).

% Numeric cite
The method \cite{smith2020} achieves ...
% Output: The method [12] achieves ...

% Print the bibliography at the end
\printbibliography[heading=bibintoc, title={References}]
```

### Build pipeline

```
pdflatex main
biber main
pdflatex main
pdflatex main
```

## Legacy: BibTeX

Stick with this only if you can't switch toolchains.

```latex
% Preamble
\usepackage{cite}    % optional: groups consecutive citations

% Body
The result \cite{smith2020, jones2021} ...

% End of document
\bibliographystyle{plain}     % or ieeetr, apalike, unsrt
\bibliography{bibliography/references}
```

Build: `pdflatex → bibtex → pdflatex → pdflatex`.

## Style Choices for ISTA

ISTA does not mandate a single style. Choose ONE consistently:

| Style key | Type | Common in |
|-----------|------|-----------|
| `ieee` | Numeric `[1]` | Engineering, CS |
| `acm-reference-format` | Numeric | Computer Science (ACM) |
| `apa` | Author-year | Social sciences, humanities |
| `numeric` | Numeric | General |
| `authoryear` | (Author, Year) | General |

For a CS / engineering thesis, **IEEE** or **ACM** are safe choices.

## BibTeX Entry Examples

```bibtex
@article{smith2020,
  author  = {Smith, John and Doe, Jane},
  title   = {A Novel Approach to {LaTeX} Theses},
  journal = {Journal of Academic Writing},
  year    = {2020},
  volume  = {15},
  number  = {3},
  pages   = {123--145},
  doi     = {10.1234/jaw.2020.123},
}

@inproceedings{jones2021,
  author    = {Jones, Alice},
  title     = {Proceedings Paper Example},
  booktitle = {Proc. of the Intl. Conf. on Something},
  year      = {2021},
  pages     = {78--90},
  publisher = {ACM},
  address   = {New York, NY, USA},
}

@book{knuth1984,
  author    = {Knuth, Donald E.},
  title     = {The {\TeX}book},
  year      = {1984},
  publisher = {Addison-Wesley},
  address   = {Reading, MA},
  isbn      = {0-201-13447-0},
}

@inbook{author2019chapter,
  author    = {Author, Some},
  title     = {Chapter Title},
  booktitle = {Edited Volume Title},
  editor    = {Editor, A. and Editor, B.},
  year      = {2019},
  publisher = {Publisher},
  pages     = {45--67},
}

@misc{website2023,
  author = {{Organisation Name}},
  title  = {Page Title},
  year   = {2023},
  url    = {https://example.com/page},
  urldate = {2026-01-15},
  note   = {Accessed: 2026-01-15},
}

@phdthesis{phdname2018,
  author      = {Author, Name},
  title       = {Title of Thesis},
  school      = {ISCTE-IUL},
  year        = {2018},
  type        = {{PhD} dissertation},
}

@mastersthesis{mscname2017,
  author = {Author, Name},
  title  = {Title of MSc Thesis},
  school = {ISCTE-IUL},
  year   = {2017},
}
```

## Citation Best Practices

- **Capitalisation in titles**: protect proper nouns and acronyms with `{}`: `{LaTeX}`, `{Bayesian}`, `{API}`.
- **Author names**: `Last, First and Last2, First2`. The `and` is mandatory between authors.
- **Page ranges**: use double dash `--` (e.g. `123--145`).
- **DOI / URL**: include DOI when available; URL only for online-only sources.
- **Consistency**: one citation style for the entire thesis.
- **Never** type a citation as `[1]` directly in text — always use `\cite{key}`.
- Keep `.bib` keys descriptive: `smith2020novel`, `acm2018guidelines`.

## Multiple `.bib` files

```latex
\addbibresource{bibliography/papers.bib}
\addbibresource{bibliography/books.bib}
\addbibresource{bibliography/online.bib}
```

## Citing within a Paragraph

```latex
The method, first introduced by \textcite{smith2020} and later
refined by \textcite{jones2021}, achieves state-of-the-art results
\parencite{evaluation2022, benchmark2023}.
```

## Verifying References

Before submission:
1. Check that every `\cite{}` resolves (no `[?]` in the PDF).
2. Run a final BibTeX/Biber pass and re-compile twice.
3. Inspect the bibliography for missing fields (especially DOI, page numbers).
4. Verify that the bibliography appears in the TOC if desired.
