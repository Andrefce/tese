# Notebook → Thesis Workflow

How to turn a Jupyter notebook of experiments into Methodology and Results
prose without losing rigour or copying code dumps.

## Reading a Notebook

- Use the available notebook tools to summarise the notebook (cell list,
  outputs, image refs). Do **not** execute or modify cells unless the user
  explicitly asks.
- Treat the notebook as the source of truth for numerical results — read the
  exact value from the displayed output rather than re-running.

## Mapping to Chapters

| Notebook artefact                              | Goes into                                      |
|------------------------------------------------|------------------------------------------------|
| Data-loading / preprocessing cells             | `\subsection{Datasets}` in Chapter 4 + Methodology preprocessing component. |
| Architecture definition (`nn.Module`, configs) | `\section{System Architecture}` and the relevant `\subsection{Component …}`. |
| Training loop, optimiser, hyper-parameters     | `\section{Implementation Details}` + a hyper-param table in an appendix. |
| Evaluation cells (Dice, Hausdorff, MAE …)      | `\section{Results}` table.                     |
| Qualitative figures (renders, heatmaps)        | `\section{Results}` figures.                   |
| Notes / markdown explanations in the notebook  | Methodology prose — paraphrase, do not lift verbatim. |

## Methodology Prose (not code dumps)

Bad:

> ```python
> for epoch in range(100):
>     for x, y in loader: ...
> ```

Good:

> The model is trained for 100 epochs with mini-batches of 8 SAX volumes,
> using AdamW (`learning rate = 1e-4`, weight decay $10^{-5}$). Each epoch
> shuffles the training split and evaluates on a held-out validation set;
> the checkpoint with the lowest validation Dice loss is retained.

A short `lstlisting` (≤ 25 lines) is fine for a *novel* algorithm or
non-trivial loss; routine PyTorch boilerplate stays out.

## Results Tables from Notebook Outputs

1. Read the metric cell output (e.g. `pandas.DataFrame.to_string()`).
2. Translate into a `booktabs` table — see
   [figures-tables-math.md](./figures-tables-math.md) for the template.
3. Quote numbers with the precision the notebook shows; do not silently
   round.
4. Bold the best value per column.
5. Reference the table from prose with `\cref{tab:...}`.

## Figures from Notebook Outputs

- Notebooks usually save figures to disk via `plt.savefig(...)`. Note the
  exact path.
- Move (or ask the user to move) the file into
  [images/](../../../../images) and reference with `\includegraphics`.
- **Never** auto-copy files between directories without asking.
- If the notebook only shows the figure inline (no `savefig`), ask the user
  to export it; do not screenshot.

## Reproducibility Block

Every Methodology chapter should report, in one paragraph or a small table:

- Hardware (e.g. *NVIDIA RTX 4090, 24 GB*).
- Software versions (Python, PyTorch, MONAI, PyTorch Geometric, CUDA).
- Random seeds for `numpy`, `torch`, and the data loader.
- Dataset split sizes (train / val / test) and any patient-level grouping.

Pull these from notebook cells (`!pip list`, `torch.__version__`,
`np.random.seed(...)`, …) — do not invent.

## Domain Vocabulary (define on first use)

LV (left ventricle), RV (right ventricle), endocardium, epicardium, SAX
(short axis), LAX (long axis), SSM (statistical shape model), PCA (principal
component analysis), GNN (graph neural network), GCN, GAT, AHA-17 (American
Heart Association 17-segment model), Dice, IoU, Hausdorff distance, MAE.

Use the form *Statistical Shape Model (SSM)* the first time, *SSM* afterwards.
