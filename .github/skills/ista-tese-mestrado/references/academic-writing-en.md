# Academic English Writing Conventions

This is a brief guide to the conventions expected in a Master's thesis written in English.

## Tone & Person

- **Formal register**: avoid contractions ("don't" → "do not"), colloquialisms, and idioms.
- **Third person or impersonal** is the academic default:
  - Preferred: *This study investigates...*, *The proposed method achieves...*, *In this work, we present...*
  - Avoid: *I think...*, *You can see that...*
- The first-person plural "we" is acceptable and common in CS/engineering, even for single-author theses (representing the author and the reader).

## Tense

| Context | Tense |
|---------|-------|
| Reporting your own work / results | Past or present (be consistent) |
| Referring to figures/tables in this document | Present (*Figure 2 shows...*) |
| Reporting prior literature | Past (*Smith (2020) proposed...*) |
| Stating general truths / definitions | Present (*The algorithm converges when...*) |
| Future work | Future (*Further studies will investigate...*) |

## British vs American English

Pick ONE and stay consistent throughout. ISCTE does not mandate a variant.

| British | American |
|---------|----------|
| analyse | analyze |
| behaviour | behavior |
| centre | center |
| modelling | modeling |
| organisation | organization |
| programme (curriculum) | program |

Configure `babel` accordingly:

```latex
\usepackage[british]{babel}        % or [american]
```

## Structure of a Paragraph

1. **Topic sentence**: states the main claim.
2. **Supporting sentences**: evidence, explanation, examples.
3. **Concluding / linking sentence**: ties to the next paragraph.

Each paragraph should cover ONE main idea. Avoid paragraphs longer than ~1 page.

## Common Connectives

| Purpose | Phrases |
|---------|---------|
| Adding | Moreover, Furthermore, In addition, Additionally |
| Contrasting | However, Nevertheless, In contrast, On the other hand |
| Causation | Therefore, Consequently, As a result, Hence, Thus |
| Sequence | First, Second, Finally, Subsequently, Next |
| Example | For instance, For example, Such as, Specifically |
| Reformulation | In other words, That is, Put differently |
| Conclusion | In summary, To conclude, Overall, In short |

Vary connectives — repetitive *"However, ... However, ..."* is jarring.

## Hedging (Academic Caution)

Avoid absolute claims unless fully proven:

- *Our results suggest that...* (not *prove*)
- *This may indicate...*
- *It is likely that...*
- *To the best of our knowledge...*

Reserve definitive language for proven results.

## Citations in Text

| Style | Example |
|-------|---------|
| Author-prominent | *Smith (2020) showed that...* (`\textcite{smith2020}`) |
| Information-prominent | *... has been shown (Smith, 2020).* (`\parencite{smith2020}`) |
| Numeric | *The method [3] achieves...* (`\cite{smith2020}`) |

Cite immediately after the claim — don't pile citations at paragraph end.

## Lists

Use lists sparingly in academic prose. Convert short lists into sentences when possible.

```latex
\begin{itemize}
  \item Each item starts with a capital and ends with a period if it is a full sentence.
  \item Otherwise, keep all items grammatically parallel and end with no punctuation (or all with semicolons).
\end{itemize}
```

Parallelism rule: all items should share the same grammatical form (all nouns, all verb phrases, all sentences, etc.).

## Numbers

- Spell out one to nine in body text: *three samples*, NOT *3 samples*.
- Use numerals for 10+ and for any measurement: *10 participants*, *3 ms*.
- Use the SI system with `siunitx`:
  ```latex
  \usepackage{siunitx}
  We measured a latency of \SI{2.5}{\milli\second}.
  ```
- Always include the unit with the value.

## Abbreviations & Acronyms

- Define on first use: *"Convolutional Neural Network (CNN)"*. Use *CNN* thereafter.
- Do not use *etc.*, *e.g.*, *i.e.* in formal academic prose — prefer *and so on*, *for example*, *that is*.
- For frequent acronyms, use the `acronym` or `glossaries` package:
  ```latex
  \usepackage{acronym}
  \acrodef{CNN}{Convolutional Neural Network}
  % Body:
  We use a \ac{CNN}.        % First: "Convolutional Neural Network (CNN)"
  Later, the \ac{CNN}.       % Subsequent: "CNN"
  ```

## Punctuation

- Use the **Oxford comma** for lists of three or more: *a, b, and c*.
- Quotation marks: use LaTeX's `` ` ` ` ``...''`` `` (not straight quotes).
- Em dash for parenthetical breaks: `---` (no spaces, gives `—`).
- Avoid exclamation marks entirely.

## Equations as Grammar

Equations are part of the sentence. Punctuate accordingly:

> Substituting into Equation~\eqref{eq:base}, we obtain
> \begin{equation}
>   f(x) = x^2 + 1,
> \end{equation}
> which is positive for all real $x$.

Note the comma after the equation and the lowercase "which".

## Common Pitfalls

- **"Data" is plural**: *the data are noisy*, not *is*.
- **"This"** alone is often ambiguous — specify: *this result*, *this approach*.
- Don't start a sentence with a symbol or number: *We find that $x = 5$*, NOT *$x = 5$ is the result*.
- Avoid passive overuse — *"It was decided that..."* hides the agent.
- Don't write "very unique" or "more optimal" — *unique* and *optimal* are absolute.

## Pre-submission Checklist

- [ ] Consistent spelling variant (British OR American)
- [ ] Consistent tense within each section
- [ ] All acronyms defined on first use
- [ ] No contractions or colloquialisms
- [ ] Each paragraph has a clear topic sentence
- [ ] Citations placed immediately after the claim
- [ ] All claims are properly hedged or supported
- [ ] Equations punctuated as part of sentences
