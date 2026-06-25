# Document Structure

Order of contents for an ISTA Master's thesis in English.

## Top-Level Skeleton

```latex
\documentclass[12pt, reqno, twoside]{amsbook}

% ... preamble (see formatting-norms.md) ...

\begin{document}

\frontmatter
  \input{frontmatter/cover}
  \input{frontmatter/dedication}
  \input{frontmatter/acknowledgments}
  \input{frontmatter/resumo}
  \input{frontmatter/abstract}
  \tableofcontents
  \listoffigures      % optional
  \listoftables       % optional

\mainmatter
  \setcounter{page}{1}
  \pagenumbering{arabic}
  \input{chapters/01-introduction}
  \input{chapters/02-literature-review}
  \input{chapters/03-methodology}
  \input{chapters/04-results}
  \input{chapters/05-conclusions}

\backmatter
  \input{bibliography/references}
  \appendix
  \input{appendices/appendix-a}

\end{document}
```

## Front Matter — Required Order

| # | Item | Required | Notes |
|---|------|----------|-------|
| 1 | **Cover page** | Yes | Official ISTA cover; `\thispagestyle{empty}` |
| 2 | Dedication | Optional | Right-aligned italic, single page |
| 3 | **Acknowledgments** | Yes | `\chapter*{Acknowledgment}` |
| 4 | **Resumo** | **Yes (mandatory)** | Portuguese abstract — required even in EN thesis |
| 5 | **Abstract** | Yes | English abstract |
| 6 | **Table of Contents** | Yes | `\tableofcontents` (depth 2) |
| 7 | List of Figures | Optional | `\listoffigures` |
| 8 | List of Tables | Optional | `\listoftables` |
| 9 | List of Acronyms / Glossary | Optional | If using many abbreviations |

### Front-matter snippets

```latex
% Dedication
\begin{dedication}
\begin{flushright}
\textit{To my family.}
\end{flushright}
\end{dedication}

% Acknowledgment
\chapter*{Acknowledgment}
\addcontentsline{toc}{chapter}{Acknowledgment}
% ... text ...

% Resumo (Portuguese)
\chapter*{Resumo}
\addcontentsline{toc}{chapter}{Resumo}
% ... Portuguese abstract ...
\noindent\textbf{Palavras-chave:} keyword1, keyword2, ...

% Abstract (English)
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}
% ... English abstract ...
\noindent\textbf{Keywords:} keyword1, keyword2, ...
```

The `\addcontentsline{toc}{chapter}{...}` line makes unnumbered chapters appear in the TOC.

## Main Matter — Typical Chapter Layout

| Chapter | Purpose |
|---------|---------|
| **Introduction** | Context, problem, research questions, objectives, contributions, document outline |
| **Literature Review / Background** | State of the art, theoretical foundation, related work |
| **Methodology / Approach** | Proposed solution, system design, methods |
| **Implementation** (optional) | Technical details, technology stack, key decisions |
| **Results / Evaluation** | Experiments, metrics, validation, comparison |
| **Discussion** (optional) | Interpretation, threats to validity, limitations |
| **Conclusions** | Summary, contributions, future work |

A typical chapter opening:

```latex
\chapter{Introduction}
\label{ch:introduction}

\noindent This chapter introduces the problem of ...

\section{Motivation}
\noindent Recent advances in ...

\section{Research Questions}
\noindent This work addresses the following questions:
\begin{enumerate}[label=\textbf{RQ\arabic*.}]
  \item How can ...
  \item What is the impact of ...
\end{enumerate}

\section{Contributions}
\noindent The main contributions of this thesis are:
\begin{itemize}
  \item A novel ...
  \item An empirical evaluation of ...
\end{itemize}

\section{Document Structure}
\noindent The remainder of this document is organised as follows.
Chapter~\ref{ch:literature-review} reviews ...
```

## Back Matter

```latex
\backmatter

% Bibliography
\renewcommand{\bibname}{References}
\bibliographystyle{plain}        % or {ieeetr}, {apalike}, etc.
\bibliography{bibliography/references}

% Appendices (optional)
\appendix
\chapter{Source Code}
\chapter{Additional Results}
```

Appendices are placed after the bibliography and labelled A, B, C... automatically via `\appendix`.

## Page Numbering Transitions

| Section | Style | Reset |
|---------|-------|-------|
| Cover, dedication | none (`\thispagestyle{empty}`) | n/a |
| Other front matter | Roman (`i, ii, iii`) | `\setcounter{page}{1}` after `\frontmatter` |
| Main matter | Arabic (`1, 2, 3`) | `\setcounter{page}{1}` after `\mainmatter` |
| Back matter | continues Arabic | no reset |
