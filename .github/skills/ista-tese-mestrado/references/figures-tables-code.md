# Figures, Tables, Code Listings, and Algorithms

## Figures

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{my-figure}
  \caption{A clear, descriptive caption that explains the figure.}
  \label{fig:my-figure}
\end{figure}
```

### Float placement

| Specifier | Meaning |
|-----------|---------|
| `h` | here (if possible) |
| `t` | top of page |
| `b` | bottom of page |
| `p` | page of floats only |
| `!` | override LaTeX's internal float rules |
| `H` | exactly here (requires `\usepackage{float}`) |

Use `[htbp]` as default. Reserve `[H]` for figures that MUST sit where placed.

### Side-by-side figures

```latex
\usepackage{subcaption}

\begin{figure}[htbp]
  \centering
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig-a}
    \caption{First view.}
    \label{fig:a}
  \end{subfigure}
  \hfill
  \begin{subfigure}[b]{0.45\textwidth}
    \centering
    \includegraphics[width=\textwidth]{fig-b}
    \caption{Second view.}
    \label{fig:b}
  \end{subfigure}
  \caption{Comparison of the two approaches.}
  \label{fig:comparison}
\end{figure}
```

Reference as `\cref{fig:a}`, `\cref{fig:comparison}`.

### Format guidelines

- Prefer **vector formats** (PDF, SVG converted to PDF) over raster (PNG/JPG).
- Place all images in `images/` and use `\graphicspath{{images/}}`.
- Don't include extension in `\includegraphics{name}` — LaTeX picks the best one.
- Captions go **below** figures, **above** tables (academic convention).
- Captions are full sentences ending with a period.

## Tables

Use `booktabs` for professional tables — never use `\hline`/vertical rules.

```latex
\usepackage{booktabs}

\begin{table}[htbp]
  \centering
  \caption{Comparison of methods on the benchmark.}
  \label{tab:comparison}
  \begin{tabular}{l c c c}
    \toprule
    Method        & Precision & Recall & F1 \\
    \midrule
    Baseline      & 0.72      & 0.68   & 0.70 \\
    Proposed      & \textbf{0.85} & \textbf{0.81} & \textbf{0.83} \\
    \bottomrule
  \end{tabular}
\end{table}
```

### Wide / multi-page tables

```latex
\usepackage{tabularx}            % auto-resize columns
\usepackage{longtable}           % tables that span pages
\usepackage{multirow}            % cells spanning multiple rows
```

```latex
\begin{tabularx}{\textwidth}{l X X}
  \toprule
  Item & Description & Notes \\
  \midrule
  A & A description that wraps. & Some notes that also wrap. \\
  \bottomrule
\end{tabularx}
```

### Best practices

- Avoid vertical rules entirely; use horizontal `\toprule`, `\midrule`, `\bottomrule`.
- Right-align numbers (`r` or use `siunitx`'s `S` column).
- Align decimal points with `siunitx`:
  ```latex
  \usepackage{siunitx}
  \begin{tabular}{l S[table-format=2.3]}
    Method & {Score} \\
    A      & 12.345  \\
    B      & 1.7     \\
  \end{tabular}
  ```

## Code Listings (CS Thesis)

### Option A: `listings` (no external dependencies)

```latex
\usepackage{listings}
\usepackage{xcolor}

\lstdefinestyle{thesisCode}{
  basicstyle      = \ttfamily\small,
  keywordstyle    = \color{blue}\bfseries,
  commentstyle    = \color{gray}\itshape,
  stringstyle     = \color{purple},
  numbers         = left,
  numberstyle     = \tiny\color{gray},
  numbersep       = 8pt,
  frame           = single,
  breaklines      = true,
  showstringspaces = false,
  tabsize         = 2,
  captionpos      = b,
}
\lstset{style=thesisCode}
```

Usage:

```latex
\begin{lstlisting}[language=Python, caption={Hello world example.}, label={lst:hello}]
def hello():
    print("Hello, thesis!")
\end{lstlisting}
```

Inline code: `\lstinline|x = 42|`.

### Option B: `minted` (better syntax, requires `-shell-escape` and Pygments)

```latex
\usepackage{minted}
\setminted{
  fontsize=\small,
  linenos,
  frame=single,
  breaklines,
}
```

Usage:

```latex
\begin{minted}[caption={Hello world.}, label=lst:hello]{python}
def hello():
    print("Hello, thesis!")
\end{minted}
```

Compile with: `pdflatex -shell-escape main.tex`.

> Use `minted` for prettier output if your build pipeline allows shell escape; otherwise stick with `listings`.

## Algorithms (Pseudocode)

```latex
\usepackage{algorithm}
\usepackage{algpseudocode}

\begin{algorithm}[htbp]
  \caption{Binary search.}
  \label{alg:binsearch}
  \begin{algorithmic}[1]
    \Require Sorted array $A[1..n]$, target $x$
    \Ensure Index $i$ such that $A[i] = x$, or $-1$
    \State $lo \gets 1$
    \State $hi \gets n$
    \While{$lo \leq hi$}
      \State $mid \gets \lfloor (lo + hi)/2 \rfloor$
      \If{$A[mid] = x$}
        \State \Return $mid$
      \ElsIf{$A[mid] < x$}
        \State $lo \gets mid + 1$
      \Else
        \State $hi \gets mid - 1$
      \EndIf
    \EndWhile
    \State \Return $-1$
  \end{algorithmic}
\end{algorithm}
```

> Note: this clashes with the template's `algorithm` theorem environment. If you use `algpseudocode`, rename the template's theorem environment (e.g. `\newtheorem{algo}{Algorithm}[chapter]`) or skip declaring it.

## Diagrams

For architecture / flow diagrams, prefer **TikZ** for native LaTeX rendering:

```latex
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

\begin{figure}[htbp]
  \centering
  \begin{tikzpicture}[node distance=2cm, every node/.style={align=center}]
    \node[draw, rectangle] (a) {Input};
    \node[draw, rectangle, right=of a] (b) {Process};
    \node[draw, rectangle, right=of b] (c) {Output};
    \draw[-Stealth] (a) -- (b);
    \draw[-Stealth] (b) -- (c);
  \end{tikzpicture}
  \caption{System pipeline.}
  \label{fig:pipeline}
\end{figure}
```

For complex diagrams created externally, export as PDF and include with `\includegraphics`.
