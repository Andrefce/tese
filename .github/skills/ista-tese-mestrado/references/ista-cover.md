# ISTA Cover Page

The cover must follow the official ISTA-IUL Master's thesis format. A ready-to-paste snippet lives in [../assets/ista-cover-page.tex](../assets/ista-cover-page.tex).

## Required Elements (in order, top to bottom)

1. **ISTA / ISCTE logo** (top, centred or left)
2. **Thesis title** (large, bold) — may include a subtitle
3. **Author full name**
4. **Degree statement**: e.g. *"Master's Thesis in [Programme Name]"*
5. **Supervisor(s)**: name, title (e.g. PhD), affiliation
6. **Co-supervisor** (if applicable)
7. **Month, Year** (e.g. *September 2026*)

## Reference Template

```latex
\thispagestyle{empty}
\begin{center}

  % --- Logo ---
  \vspace*{1cm}
  \includegraphics[width=0.45\textwidth]{images/ista-logo}\\[2cm]

  % --- Title ---
  {\LARGE\bfseries Title of the Thesis}\\[0.5cm]
  {\large Optional Subtitle of the Thesis}\\[2cm]

  % --- Author ---
  {\large Author Full Name}\\[2cm]

  % --- Degree ---
  {\normalsize Master's Thesis in}\\[0.3cm]
  {\large\bfseries Programme Name}\\[2cm]

  % --- Supervisors ---
  {\normalsize Supervisor:}\\
  {\normalsize Prof. Supervisor Name, PhD}\\
  {\normalsize Institution / Department}\\[0.5cm]

  {\normalsize Co-Supervisor:}\\
  {\normalsize Prof. Co-Supervisor Name, PhD}\\
  {\normalsize Institution / Department}\\[2cm]

  % --- Date ---
  {\normalsize Month, Year}

  \vfill
\end{center}
\clearpage
```

## Logo File

Place the official logo in `images/ista-logo.{pdf,png}`. Prefer **PDF** (vector) over PNG.

If using `graphicspath`:

```latex
\usepackage{graphicx}
\graphicspath{{images/}}
```

Then reference simply as `\includegraphics[...]{ista-logo}`.

## Common Mistakes

- Wrong logo (using outdated ISCTE logo instead of current ISTA-IUL one).
- Indented content on cover (cover must be `\begin{center}` and not affected by `\noindent` rules).
- Page number visible on cover (must use `\thispagestyle{empty}`).
- Wrong degree wording — confirm exact programme name with the institution.
- Adding cover to the TOC (must NOT appear in `\tableofcontents`).

## Quick Compliance Checklist

- [ ] `\thispagestyle{empty}` is set
- [ ] ISTA logo present (vector format)
- [ ] Title is the largest text on the page
- [ ] Full author name (no abbreviations)
- [ ] Correct programme name
- [ ] Supervisor(s) with title and affiliation
- [ ] Month and Year only (no day)
- [ ] `\clearpage` after the cover
