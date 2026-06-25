# Cross-References

Use `cleveref` for smart, automatic cross-references that produce text like "Figure 2.3" or "Section 4.1" without typing the prefix manually.

## Setup

```latex
\usepackage{hyperref}                  % must be loaded BEFORE cleveref
\usepackage[noabbrev, capitalise]{cleveref}
```

| Option | Effect |
|--------|--------|
| `noabbrev` | "Equation" instead of "Eq." |
| `capitalise` | Always capitalise (e.g. "Figure" not "figure") |
| `nameinlink` | Make the name part of the hyperlink |

## Label Conventions

Use a consistent prefix for each kind of label. This makes `\cref` work correctly and the source easier to read.

| Object | Prefix | Example |
|--------|--------|---------|
| Chapter | `ch:` | `\label{ch:introduction}` |
| Section | `sec:` | `\label{sec:methodology}` |
| Subsection | `subsec:` | `\label{subsec:setup}` |
| Equation | `eq:` | `\label{eq:einstein}` |
| Figure | `fig:` | `\label{fig:pipeline}` |
| Subfigure | `fig:` | `\label{fig:pipeline-a}` |
| Table | `tab:` | `\label{tab:results}` |
| Algorithm | `alg:` | `\label{alg:binsearch}` |
| Listing (code) | `lst:` | `\label{lst:hello}` |
| Theorem | `thm:` | `\label{thm:main}` |
| Lemma | `lem:` | `\label{lem:helper}` |
| Definition | `def:` | `\label{def:metric}` |
| Appendix | `app:` | `\label{app:dataset}` |

## Reference Commands

| Command | Output |
|---------|--------|
| `\ref{fig:x}` | `2.3` |
| `\pageref{fig:x}` | `47` |
| `\cref{fig:x}` | `Figure 2.3` |
| `\Cref{fig:x}` | `Figure 2.3` (start of sentence) |
| `\cref{fig:x,fig:y}` | `Figures 2.3 and 2.4` |
| `\cref{fig:x,fig:y,fig:z}` | `Figures 2.3 to 2.5` |
| `\cref{eq:einstein}` | `Equation (2.1)` |
| `\eqref{eq:einstein}` | `(2.1)` (without prefix word) |
| `\cref{sec:method}` | `Section 4.1` |

## Examples

```latex
% Capitalise at start of sentence:
\Cref{fig:pipeline} shows the overall architecture.

% Mid-sentence — lowercase is automatic:
We compare against the baseline in \cref{tab:results}.

% Multiple refs:
\Cref{fig:a,fig:b} show two perspectives of the same system.

% Range:
The results are presented in \cref{tab:res1,tab:res2,tab:res3}.
% Output: Tables 5.1 to 5.3

% Page reference:
The detailed proof is on \cpageref{thm:main}.
```

## When Not to Use cleveref

For equations, `\eqref{}` from `amsmath` is still idiomatic and produces `(2.1)`:

```latex
By \eqref{eq:einstein}, we obtain ...
```

Both work — pick a style and be consistent.

## Forward References

LaTeX needs **two compilations** to resolve forward references:
1. First pass: writes labels to `.aux` file
2. Second pass: reads `.aux` and substitutes references

If you see `??` in the PDF, recompile.

## Tips

- Define the label IMMEDIATELY after the caption / numbered environment, not before.
- For equations, the label can go anywhere inside `equation`/`align`, but right before the closing tag is conventional.
- Never type "Figure 3" manually — always `\cref{fig:...}`. This survives renumbering.
- If a label is missing, LaTeX prints `[?]` and a warning. Search the build log for `LaTeX Warning: Reference` to find undefined refs.
