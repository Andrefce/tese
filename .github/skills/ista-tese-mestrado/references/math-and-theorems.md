# Math, Theorems, and Equations

## Theorem-like Environments

The template defines (chapter-numbered):

| Environment | Use for |
|-------------|---------|
| `theorem` | Main results |
| `lemma` | Supporting results used to prove theorems |
| `proposition` | Smaller standalone results |
| `corollary` | Direct consequences |
| `definition` | Formal definitions |
| `example` | Worked examples |
| `remark` | Side comments |
| `proof` | Proofs (unnumbered, ends with $\square$) |
| `algorithm` | Algorithm statements (text-style) |
| `conjecture` | Open conjectures |
| `notation` | Notational conventions |
| `claim` | Sub-claims within proofs |
| `axiom`, `criterion`, `condition`, `problem`, `solution`, `exercise`, `case`, `conclusion`, `summary`, `acknowledgement` | Less common, available |

### Usage

```latex
\begin{theorem}[Optional Name]
  \label{thm:main}
  Let $H$ be a Hilbert space and ...
\end{theorem}

\begin{proof}
  By assumption, ...
  \begin{equation}
    \|x - y\| \leq \varepsilon. \label{eq:bound}
  \end{equation}
  Combining \eqref{eq:bound} with \cref{lem:helper} yields the result.
\end{proof}
```

### Adding new theorem-like environments

```latex
\theoremstyle{plain}        % bold name, italic body  (default for theorem/lemma)
\theoremstyle{definition}   % bold name, upright body (recommended for definition/example)
\theoremstyle{remark}       % italic name, upright body

\newtheorem{hypothesis}{Hypothesis}[chapter]   % H 1.1, H 2.1 ...
```

## Equations

### Inline math

Use `$...$` (preferred) — never `$$...$$` for display math.

```latex
The function $f(x) = x^2$ is continuous.
```

### Numbered display equations

```latex
\begin{equation}
  E = mc^2 \label{eq:einstein}
\end{equation}
```

### Unnumbered display

```latex
\begin{equation*}
  a^2 + b^2 = c^2
\end{equation*}
% or
\[ a^2 + b^2 = c^2 \]
```

### Aligned equations

```latex
\begin{align}
  f(x) &= (x+1)(x-1) \label{eq:expand1}\\
       &= x^2 - 1. \label{eq:expand2}
\end{align}
```

For a single equation number across multiple lines, use `aligned` inside `equation`:

```latex
\begin{equation}
  \begin{aligned}
    f(x) &= (x+1)(x-1) \\
         &= x^2 - 1.
  \end{aligned}
  \label{eq:expand}
\end{equation}
```

### Cases

```latex
f(x) =
\begin{cases}
  0 & \text{if } x < 0, \\
  1 & \text{if } x \geq 0.
\end{cases}
```

### Matrices

```latex
\begin{pmatrix}
  a & b \\
  c & d
\end{pmatrix}
```

Variants: `bmatrix` (square), `vmatrix` (single bars, determinant), `Vmatrix` (double bars, norm), `matrix` (no delimiters).

## Referencing Equations

| Command | Output |
|---------|--------|
| `\eqref{eq:einstein}` | `(2.1)` |
| `\ref{eq:einstein}` | `2.1` |
| `\cref{eq:einstein}` | `Equation~(2.1)` (with cleveref) |
| `\Cref{eq:einstein}` | `Equation~(2.1)` at start of sentence |

Prefer `\cref` from the `cleveref` package — it adds the prefix automatically.

## Math Style Best Practices

- Punctuate equations as part of the sentence (`. , ;` at the end).
- Use `\,` for thin space before differentials: `\int f(x)\,dx`.
- Use `\operatorname{name}` for custom operators: `\operatorname{argmin}`.
- Use `\text{...}` inside math for text: `$x \in A \text{ such that } x > 0$`.
- Use `\mathbb{R}, \mathbb{N}, \mathbb{Z}` for number sets.
- Use `\mathcal{O}` for big-O notation: $f(n) = \mathcal{O}(n \log n)$.
- Vectors: pick one convention — `\mathbf{x}` (bold) OR `\vec{x}` (arrow) — stay consistent.

## Common Pitfalls

- Multi-letter variables in italic look like products: write `\mathit{rate}` or `\text{rate}` instead of `rate`.
- Always wrap units in `\text{}` or use `siunitx`: `\SI{3.5}{\meter\per\second}`.
- For Greek capital letters that look upright: `\varGamma`, `\varDelta` (italic variants).
- Never use `\over` — use `\frac{}{}` instead.
- Never use `$$...$$` — use `\[...\]` or `equation*`.
