# Formatting Norms (Preamble & Layout)

ISTA-IUL Master's thesis formatting rules and the LaTeX preamble that enforces them.

## Document Class

```latex
\documentclass[12pt, reqno, twoside]{amsbook}
```

| Option | Effect |
|--------|--------|
| `12pt` | Base font size — required |
| `reqno` | Equation numbers on the right |
| `twoside` | Different margins for odd/even pages (for print binding) |

> Use `oneside` only for the digital-only version if asked.

## Required Packages (canonical preamble)

```latex
% Math
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{amsthm}

% Encoding & language
\usepackage[utf8]{inputenc}        % source encoding
\usepackage[T1]{fontenc}           % output encoding (for accented chars)
\usepackage[english]{babel}        % English hyphenation/conventions

% Layout
\usepackage[a4paper, margin=2.5cm]{geometry}
\usepackage[onehalfspacing]{setspace}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{chngcntr}

% Graphics & misc
\usepackage{graphicx}
\usepackage{eurosym}
\usepackage{enumitem}
\usepackage{etoolbox}

% Cross-references & links (load LAST among these)
\usepackage[hidelinks]{hyperref}   % clickable refs/TOC
\usepackage[noabbrev]{cleveref}    % smart \cref
```

### Optional but recommended

```latex
\usepackage{booktabs}              % nicer tables
\usepackage{microtype}             % better justification
\usepackage{csquotes}              % proper quotation marks
\usepackage{xcolor}                % colours
% \usepackage{epstopdf}            % only if you use EPS figures
```

## Page Geometry

- **Paper**: A4
- **Margins**: 2.5 cm top/bottom/left/right
- **Binding offset**: not required by default; add `bindingoffset=1cm` only if printed and bound

```latex
\usepackage[a4paper, margin=2.5cm]{geometry}
% Or with binding offset:
% \usepackage[a4paper, margin=2.5cm, bindingoffset=1cm]{geometry}
```

## Line Spacing

```latex
\usepackage[onehalfspacing]{setspace}   % 1.5 spacing
```

For specific regions of double or single spacing use `\begin{singlespace}...\end{singlespace}`.

## Headers & Footers

```latex
\fancyhead{}
\fancyfoot{}
\pagestyle{fancy}
\fancyfoot[LE,RO]{\thepage}     % page no.: left-even, right-odd
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
```

For the chapter title in headers (optional):

```latex
\fancyhead[LE]{\nouppercase{\leftmark}}   % chapter on even pages
\fancyhead[RO]{\nouppercase{\rightmark}}  % section on odd pages
\renewcommand{\headrulewidth}{0.4pt}
```

## Section Styling

The template uses bold flush-left section titles:

```latex
\makeatletter
  \def\section{\@startsection{section}{1}%
    \z@{.5\linespacing\@plus.7\linespacing}{.25\linespacing}%
    {\normalfont\bfseries\flushleft}}
  \def\subsection{\@startsection{subsection}{2}%
    \z@{.5\linespacing\@plus.7\linespacing}{.25\linespacing}%
    {\normalfont\bfseries\flushleft}}
\makeatother
```

## Numbering Scopes

```latex
\numberwithin{equation}{chapter}    % (1.1), (2.3) ...
\numberwithin{section}{chapter}     % 1.1, 1.2, 2.1 ...
\numberwithin{figure}{chapter}      % Figure 1.1, 2.4 ...
\numberwithin{table}{chapter}       % Table 1.1, 3.2 ...
```

## Paragraph Indentation Rule

In `amsbook`, first paragraphs of sections are NOT indented automatically. The template authors require manual `\noindent`:

```latex
\section{Motivation}
\noindent The motivation for this work ...
```

Subsequent paragraphs are indented automatically — do NOT add `\noindent` to those.

## Fonts

The default `amsbook` font (Computer Modern) is acceptable. For a more modern look:

```latex
\usepackage{lmodern}               % Latin Modern (improved Computer Modern)
% or
\usepackage{newtxtext, newtxmath}  % Times-like
```

Note: ISTA does not mandate a specific font for the LaTeX version. Default Computer Modern is fine for submission.

## Hyperref Settings

```latex
\hypersetup{
  pdftitle    = {Title of the Thesis},
  pdfauthor   = {Author Name},
  pdfsubject  = {Master's Thesis, ISTA-IUL},
  pdfkeywords = {keyword1, keyword2},
  hidelinks                    % suppress coloured boxes around links
}
```

Always load `hyperref` AFTER most other packages and BEFORE `cleveref`.
