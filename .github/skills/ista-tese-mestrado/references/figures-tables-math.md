# Figures, Tables, Math, Code

Hard rules and copy-paste templates.

## Figures

- Place files in [images/](../../../../images) (already on
  `\graphicspath{{images/}}` from `main.tex`).
- Caption goes **below** the graphic; label goes **after** the caption.
- Refer with `\cref{fig:...}`, never `Figure~\ref{...}`.

```latex
\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\textwidth]{lv-pipeline}
  \caption{Overview of the LV reconstruction pipeline: SAX slices are
  segmented, the contours are stacked into a sparse point cloud, and a GNN
  refines the resulting mesh.}
  \label{fig:lv-pipeline}
\end{figure}
```

For two side-by-side images, use `subcaption` (already implicitly fine via
`graphicx`; load `\usepackage{subcaption}` in `main.tex` only when you need
it).

## Tables

- Use `booktabs` rules (`\toprule` / `\midrule` / `\bottomrule`); never
  `\hline` and never vertical bars.
- Caption **above** the table.
- Numerical columns: align on the decimal with `S` from `siunitx` if you load
  it; otherwise right-align with `r`.
- Bold the best value per metric.

```latex
\begin{table}[H]
  \centering
  \caption{Wall-thickness mean absolute error (mm) on the test split.}
  \label{tab:thickness-mae}
  \begin{tabular}{lccc}
    \toprule
    Method                       & Apex & Mid & Base \\
    \midrule
    SSM baseline                 & 1.42 & 1.18 & 1.05 \\
    U-Net + nearest-neighbour    & 1.10 & 0.95 & 0.88 \\
    Proposed (SSM + GNN)         & \textbf{0.86} & \textbf{0.74} & \textbf{0.69} \\
    \bottomrule
  \end{tabular}
\end{table}
```

## Equations

- Single equation: `equation` environment with a `\label{eq:...}`.
- Multi-line: `align`, one label per numbered line (or `\nonumber` for the
  rest).
- Refer with `\cref{eq:...}`, not `(\ref{...})`.

```latex
\begin{equation}
  d(\mathbf{x}) = \min_{\mathbf{y} \in \mathcal{S}_{\text{epi}}}
                  \lVert \mathbf{x} - \mathbf{y} \rVert_2,
  \label{eq:thickness}
\end{equation}
```

```latex
\begin{align}
  \mathcal{L}_{\text{recon}} &= \frac{1}{N}\sum_{i=1}^{N} \lVert v_i - \hat{v}_i \rVert_2^2,
  \label{eq:recon-loss} \\
  \mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{recon}}
                                + \lambda\, \mathcal{L}_{\text{reg}}.
  \label{eq:total-loss}
\end{align}
```

## Code Listings

Use the preconfigured `thesisCode` style (`\lstset{style=thesisCode}` is
already in `main.tex`). Always set `caption=` and `label=`.

```latex
\begin{lstlisting}[language=Python,
                   caption={Pointwise wall-thickness from endo/epi meshes.},
                   label={lst:thickness}]
def wall_thickness(endo_pts, epi_tree):
    dist, _ = epi_tree.query(endo_pts)
    return dist
\end{lstlisting}
```

For listings longer than ~25 lines, push them to an appendix and reference
with `\cref{lst:...}`.

## Cross-references

| Target              | Use                             |
|---------------------|---------------------------------|
| Chapter             | `\cref{ch:results}`             |
| Section             | `\cref{sec:setup}`              |
| Subsection          | `\cref{subsec:datasets}`        |
| Figure              | `\cref{fig:lv-pipeline}`        |
| Table               | `\cref{tab:thickness-mae}`      |
| Equation            | `\cref{eq:thickness}`           |
| Listing             | `\cref{lst:thickness}`          |
| Appendix            | `\cref{app:appendix-a}`         |

`\Cref{...}` capitalises the prefix when starting a sentence.
