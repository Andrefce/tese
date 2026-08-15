# Thesis Review — Action Checklist

This checklist contains all the issues identified in the thesis review.  
For each item, the **Action / Decision** field is intentionally left blank so I can state what should be done.

---

WHEN say use simpler language, it means that the text should be rewritten in a way that is more accessible to a general scientific audience. Avoid overly technical jargon, complex sentence structures, and unnecessary verbosity. The goal is to make the content clear and understandable without losing scientific accuracy. also dont use bullet points and the '-' symbol, just write in simple sentences. Also dont say we, use neutral language. Avoid using first-person pronouns like "we" or "I". Instead, use neutral language that focuses on the research and findings rather than the authors. For example, instead of saying "We conducted an experiment," say "An experiment was conducted." This helps maintain an objective and professional tone throughout the thesis. Im not a native English speaker, so please make sure to use simple and clear language that is easy to understand. Avoid complex sentence structures and technical jargon that may be difficult for non-native speakers to comprehend. The goal is to ensure that the content is accessible and understandable to a wider audience, including those who may not have a strong background in the subject matter.

---

# 🔴 CRITICAL — MUST FIX

## 1. Resolve Table 4.1 vs. surrounding prose contradictions

**Problem:** The prose reports substantially different reconstruction metrics from Table 4.1.

Reported in prose:
- Endocardial Dice: **0.940 ± 0.010**
- Myocardium Dice: **0.843 ± 0.018**
- F-score @ 1 mm: **0.610 ± 0.044**
- F-score @ 2 mm: **0.897 ± 0.026**
- Cavity volume ratio: **0.929**
- Epicardial volume ratio: **1.012**
- Myocardial volume ratio: **1.126**

Table 4.1 reports:
- Endocardium Dice: **0.86 ± 0.03**
- Myocardium Dice: **0.66 ± 0.07**
- Endocardium IoU: **0.76 ± 0.05**
- Myocardium IoU: **0.50 ± 0.08**
- F-score @ 1 mm: **0.25 ± 0.06**
- F-score @ 2 mm: **0.49 ± 0.10**
- Cavity volume ratio: **0.98 ± 0.06**
- Epicardial volume ratio: **1.02 ± 0.05**
- Myocardial volume ratio: **1.09 ± 0.12**

**Action / Decision:** SEE THE NEW RESULTS WITH THE NEW ARCHITECTURE AND UPDATE THE TABLE AND PROSE TO MATCH.


---

## 2. Resolve Table 4.3 vs. Discussion contradictions

**Problem:** Table 4.3 and the later discussion report different ICC/bias/limits-of-agreement values.

Table 4.3:
- Laplace ICC: **0.76**
- Yezzi–Prince ICC: **0.76**
- SDF cone rays ICC: **0.16**
- EDT ICC: **0.78**
- Laplace bias: **+0.62 mm**
- Laplace LoA: approximately **±1.87 mm**
- Cone-ray LoA: approximately **±8.13 mm**

Discussion later states:
- Field-based estimators: ICC approximately **0.52–0.57**
- Cone-ray ICC approximately **0.15**
- Cone-ray limits approximately **±5.07 mm**

**Action / Decision:** SEE THE NEW RESULTS WITH THE NEW ARCHITECTURE AND UPDATE THE TABLE AND PROSE TO MATCH.


---

## 3. Resolve Table 4.4 vs. Discussion percentage contradiction

**Problem:** The thesis reports different percentage differences in wall thickness between model and segmentation-derived reference.

One location reports approximately **+39%**.

Another discussion reports approximately **+52.9%**.

**Action / Decision:** SEE THE NEW RESULTS WITH THE NEW ARCHITECTURE AND UPDATE THE TABLE AND PROSE TO MATCH.


---

## 4. Resolve the 400 vs. 776 epochs contradiction

**Problem:** Methodology states:

> Maximum 400 epochs per stage.

Results state:

> Fine-tuning runs for 776 epochs.

**Action / Decision:** LEAVE IT FOR NOW


---

## 5. Resolve mesh post-processing contradiction

**Problem:** Methodology says:

> "No mesh repair or hole filling is applied afterwards; the surfaces come directly out of the model."

But the Results describe:
- degenerate-face removal
- largest-component selection
- hole filling
- remeshing through a cleaned occupancy mask

The Discussion also acknowledges that raw level sets can contain handles and require repair.

**Action / Decision:** UPDATE TO THE ACTUAL POST PROCESSING USED, ON 3.4.4 of methodology update with simple language what was done and why it was necessary. Then update the results and discussion to match.


---

## 6. Remove placeholder / estimated experimental results

**Problem:** The pre-training section explicitly contains "placeholder figures" and approximate/expected numerical values.

Examples include approximate loss values such as:
- about **6.4 → 1.3**
- approximately **0.99**

These should not appear as experimental observations unless they came from actual recorded experiments.

**Action / Decision:** REMOVE AND SAY THAT PRE-Training occured on the first 200 epochs and the rest was fine tune (the graph show both +re training and fine tuning) and then update the results and discussion with the actual numbers from the final evaluation. (DO NOT MENTION THE NEW 50 EPOCHS OF NEW ARCHITECTURE, KEEP EVERYTHING AS IT WAS ALWAYS DONE WITH NEW ARCHITECTURE!!!)


---

## 7. Resolve M&Ms vs. M&Ms-2 dataset usage

**Problem:** Different sections describe the real training data differently.

Some sections state:
- ACDC
- M&Ms
- M&Ms-2

Other sections state:
- ACDC
- M&Ms-2

The thesis needs one definitive dataset description.

**Action / Decision:** RESOLVE MISMATCH


---

## 8. Complete the Introduction

**Problem:** The current Introduction still contains placeholders:

- RQ1
- RQ2
- RQ3
- empty Objectives
- empty Contributions

**Action / Decision:** LEAVE IT FOR NOW


---

## 9. Complete the Abstract

**Problem:** The Abstract currently contains placeholder text such as:

> "This thesis addresses the problem of..."

**Action / Decision:** LEAVE IT FOR NOW


---

## 10. Replace all thesis-template placeholders

**Problem:** The title page still contains template fields such as:

- Title of the Thesis
- Optional Subtitle of the Thesis
- Author Full Name
- Programme Name
- Prof. Supervisor Name
- Position, Department
- Institution
- Month, Year

**Action / Decision:** LEAVE IT FOR NOW


---

# 🟠 IMPORTANT SCIENTIFIC / METHODOLOGICAL ISSUES

## 11. Correct the M&Ms dataset description/count

**Problem:** The thesis states that M&Ms contains **345 subjects**, while the published M&Ms dataset is generally described as containing **375 CMR datasets**.

If 345 is a specific subset actually used by the thesis, this must be explicitly explained.

**Action / Decision:** DELETE QUALITY GATES FOR THE REAL DATA AND JUST USE THE OFFICIAL NUMBERS! UPDATE METHODOLOGY AND RESULTS TO MATCH THE OFFICIAL NUMBERS. 


---

## 12. Stop using overly absolute claims

Potential examples include:

- "universally adopted"
- "gold standard"
- "de facto backbone"
- similarly absolute statements

These should generally be replaced by appropriately qualified language such as:
- "widely adopted"
- "commonly used"
- "reference approach"
- "widely used baseline"

**Action / Decision:** USE LANGUAGE THAT IS NOT ABSOLUTE, BUT STILL STRONG. DO NOT USE "universally adopted" OR "gold standard". USE ALSO SIMPLE LANGUAGE TO EXPLAIN THE REASON WHY IT IS USED AND WHY IT IS IMPORTANT.


---

## 13. Remove or substantiate the "sub-pixel accuracy" claim

**Problem:** The thesis contains a claim that U-Net produces contours with "sub-pixel accuracy."

This is not something that follows inherently from U-Net and should either be properly supported by a specific source or removed.

**Action / Decision:** 


---

## 14. Distinguish segmentation reference from 3D ground truth

**Problem:** The thesis constructs 3D reference geometry from 2D segmentation labels.

Therefore, the terminology should be consistent.

Preferred terminology could be:
- "expert segmentation reference"
- "segmentation-derived reference geometry"
- "reference geometry derived from the segmentation"

Avoid calling the reconstructed 3D mesh itself "3D ground truth" unless there is genuine independent 3D ground truth.

**Action / Decision:** USE CONSISTENT TERMINOLOGY THROUGHOUT THE THESIS. AVOID CALLING THE RECONSTRUCTED 3D MESH "3D GROUND TRUTH". SAY THAT THERE ISNT NO DATASET PUBLISHED WITH 3D GROUND TRUTH, AND THAT THE RECONSTRUCTED 3D MESH IS A REFERENCE DERIVED FROM THE SEGMENTATION.


---

## 15. Tone down the research-gap claim

**Problem:** The thesis claims that few/no studies combine the complete proposed pipeline.

The literature review does not conclusively prove that no previous work has done so.

Prefer a formulation such as:

> "To the best of our knowledge, limited work has..."

rather than an absolute claim such as:

> "No previous studies have..."

**Action / Decision:** TONE DOWN THE CLAIM TO REFLECT LIMITED WORK RATHER THAN ABSOLUTE NOVELTY. USE SIMPLE LANGUAGE TO EXPLAIN THE REASON WHY IT IS IMPORTANT AND WHY IT IS A NOVEL CONTRIBUTION.


---

## 16. Address the repeated-measures structure of the AHA-17 analysis

**Problem:** The evaluation contains approximately:

20 patients × 17 AHA segments = 340 segment observations.

These 340 observations are not independent because multiple segments belong to the same patient.

This should be acknowledged when interpreting:
- correlation
- ICC
- confidence intervals
- statistical significance

A mixed-effects or patient-level analysis could be considered if appropriate.

**Action / Decision:** SINCE ITS THE SAME DATA FOR COMPARING THE RESULTS OF EACH MODEL, THINK OF A BETTER WAY TO DESCRIBE MAYBE, I DONT THINK USING THE MEAN VALUE FOR EACH SEGMENT IS THAT BAD SINCE ITS THE SAME DATA, BUT SEE IF THERES A BETTER WAY TO DESCRIBE IT. USE SIMPLE LANGUAGE TO EXPLAIN THE REASON WHY IT IS IMPORTANT AND WHY IT IS A NOVEL CONTRIBUTION.


---

## 17. Clarify the meaning of "positive wall thickness by construction"

**Problem:** The model parameterisation guarantees:

\[
\delta(x) \geq \tau_{\min} > 0
\]

This guarantees a positive offset mathematically.

It does **not** guarantee:
- anatomically correct wall thickness
- correct epicardial topology
- correct transmural direction
- clinically realistic thickness
- absence of discretisation artefacts

The wording should make this distinction explicit.

**Action / Decision:** USE SIMPLE LANGUAGE TO DESCRIBE THIS AND WHY IT IS NEEDED AND WHAT IT DOES.


---

## 18. Treat the ED/ES difference as a real scientific finding

**Problem / Observation:** The model appears to perform better at ED than ES, with systematic ES underestimation.

Reported observations include:
- cavity ratio around **0.90**
- epicardial ratio around **0.81**
- myocardial shell ratio around **0.77**
- wall thickness approximately **1.6 mm below** the segmentation-derived reference
- reduced systolic thickening

This should be presented as an important limitation/finding rather than hidden.

**Action / Decision:** AFTER MATCHING THE RESULTS TO THE NEW ARCHITECTURE, UPDATE THE RESULTS AND DISCUSSION TO REFLECT THIS FINDING. USE SIMPLE LANGUAGE TO DESCRIBE THIS, AND WHAT HAPPENED.


---

## 19. Be cautious with the size of the evaluation cohort

**Problem:** Reconstruction evaluation uses approximately **20 normal patients**.

The HCM analysis uses approximately **10 patients**.

This is acceptable for a Master's-level exploratory evaluation, but insufficient for strong claims of:
- clinical generalisation
- pathology generalisation
- clinical utility

Use wording such as:
> "preliminary evaluation"

where appropriate.

**Action / Decision:** MAKE AN EXCUSE FOR THIS, AND THAT IT JUST PRE LIMINARY EVALUATION, AND THAT THE RESULTS ARE PROMISING, BUT NEED TO BE VALIDATED IN A LARGER COHORT WITH ACTUAL 3D GROUND TRUTH.


---

## 20. Clarify the limitation of the synthetic epicardium

**Problem:** Synthetic epicardial geometry is generated by applying an artificial thickness offset to the endocardial geometry.

The thesis already acknowledges that this is not intended to reproduce clinical wall thickness.

This limitation should remain explicit.

The synthetic data should be described as a **geometric pretraining prior**, not as realistic clinical wall-thickness ground truth.

**Action / Decision:** MAKE SURE TO CLARIFY THAT THE SYNTHETIC EPICARDIUM IS USED FOR GEOMETRIC PRETRAINING ONLY, AND NOT AS CLINICAL GROUND TRUTH. USE SIMPLE LANGUAGE TO EXPLAIN THIS LIMITATION.


---

# 🟡 LITERATURE / REFERENCES

## 21. Verify every bibliography entry against the original publication

**Problem:** The bibliography is broadly complete, but at least some author metadata appears questionable.

Example:
- Campello et al. / M&Ms reference should be checked against the actual published author list.

Do not manually trust the current bibliography.

**Action / Decision:** VERIFY EACH BIBLIOGRAPHY ENTRY AGAINST THE ORIGINAL PUBLICATION. CORRECT ANY ERRORS IN AUTHOR NAMES, TITLES, JOURNALS, YEARS, AND DOIS. USE SIMPLE LANGUAGE TO EXPLAIN ANY CHANGES MADE.


---

## 22. Verify every DOI

**Problem:** Every reference with a DOI should be checked against:
- the actual publication
- correct DOI
- correct title
- correct journal/conference
- correct year
- correct author list

**Action / Decision:** VERIFY EACH DOI AGAINST THE ORIGINAL PUBLICATION. CORRECT ANY ERRORS IN DOI, TITLE, JOURNAL/CONFERENCE, YEAR, AND AUTHOR LIST. USE SIMPLE LANGUAGE TO EXPLAIN ANY CHANGES MADE.


---

## 23. Check citation-to-reference completeness

**Task:** Verify that:
1. Every in-text citation appears in the bibliography.
2. Every bibliography entry is actually cited somewhere.
3. There are no duplicated references under slightly different names.
4. Author/year combinations are consistent.

**Action / Decision:** CHECK THAT EVERY IN-TEXT CITATION HAS A CORRESPONDING BIBLIOGRAPHY ENTRY AND THAT THERE ARE NO DUPLICATES. USE SIMPLE LANGUAGE TO EXPLAIN ANY CHANGES MADE.


---

## 24. Check the ACDC citation

**Problem:** Ensure the ACDC dataset description cites the correct source and that the dataset characteristics stated in the thesis match the source.

**Action / Decision:** CHECK THAT THE ACDC DATASET DESCRIPTION CITES THE CORRECT SOURCE AND THAT THE DATASET CHARACTERISTICS MATCH THE SOURCE. USE SIMPLE LANGUAGE TO EXPLAIN ANY CHANGES MADE.


---

## 25. Check the UK Digital Heart / SSM citation

**Problem:** Ensure that the UK Digital Heart / statistical shape model claims are supported by the correct Bai et al. reference and that the thesis distinguishes the original atlas from the specific SSM implementation used.

**Action / Decision:** CHECK THAT THE UK DIGITAL HEART / STATISTICAL SHAPE MODEL CLAIMS ARE SUPPORTED BY THE CORRECT BAI ET AL. REFERENCE AND THAT THE THESIS DISTINGUISHES THE ORIGINAL ATLAS FROM THE SPECIFIC SSM IMPLEMENTATION USED. 


---

## 26. Check the M&Ms citation

**Problem:** Ensure the M&Ms dataset description, number of subjects, vendors, centres, countries, and segmentation information are all consistent with the cited paper.

**Action / Decision:** CHECK THAT THE M&MS DATASET DESCRIPTION, NUMBER OF SUBJECTS, VENDORS, CENTRES, COUNTRIES, AND SEGMENTATION INFORMATION ARE CONSISTENT WITH THE CITED PAPER.


---

## 27. Check the M&Ms-2 citation

**Problem:** Ensure M&Ms-2 is correctly cited and clearly distinguished from the original M&Ms dataset.

**Action / Decision:** CHECK THAT M&MS-2 IS CORRECTLY CITED AND CLEARLY DISTINGUISHED FROM THE ORIGINAL M&MS DATASET.


---

## 28. Check the DeepSDF citation and claim

**Problem:** Ensure the thesis does not attribute claims to DeepSDF that are not actually supported by Park et al. 2019.

**Action / Decision:**  CHECK
 

---

## 29. Check the Fourier-feature citation

**Problem:** Ensure the claim about Fourier positional encoding and high-frequency representation is appropriately supported by Tancik et al. 2020.

**Action / Decision:** CHECK


---

## 30. Check the Laplace-thickness citation

**Problem:** Ensure the formulation and claims attributed to Jones et al. 2000 accurately reflect the original method.

**Action / Decision:** CHECK


---

## 31. Check the Yezzi–Prince citation

**Problem:** Ensure the PDE formulation and interpretation attributed to Yezzi & Prince 2003 accurately reflect the original paper.

**Action / Decision:** CHECK


---

## 32. Check the AHA-17 citation

**Problem:** Ensure the AHA-17 segmentation model description is accurately attributed to Cerqueira et al. 2002 and avoid claims such as "universally adopted."

**Action / Decision:** CHECK


---

# 🔵 ARCHITECTURE / METHODS

## 33. Make the model pipeline explicit

The central pipeline should be clearly and consistently described as:

Sparse SAX contours  
→ point/feature encoder  
→ global latent representation + local spatial features  
→ implicit decoder  
→ endocardial SDF  
→ positive wall-thickness offset  
→ epicardial field  
→ Marching Cubes  
→ reconstructed LV meshes  
→ thickness estimation  
→ AHA-17 regional analysis

**Action / Decision:** CHECK


---

## 34. Clearly distinguish raw model output from post-processed geometry

Define two stages:

**Raw model output**
- SDF level sets
- Marching Cubes
- no repair

**Evaluation geometry**
- degenerate-face removal
- component selection
- hole filling
- remeshing, if actually performed

Do not imply that the neural architecture itself guarantees watertight topology if post-processing is required.

**Action / Decision:** DO THIS VERY CAREFULLY AND USE SIMPLE LANGUAGE TO EXPLAIN THE TWO STAGES AND WHY POST-PROCESSING IS NEEDED.


---

## 35. Clarify what the wall-thickness decoder actually guarantees

The model guarantees a positive offset:

\[
\delta(x) > 0
\]

but not necessarily correct anatomical thickness.

Explain this explicitly.

**Action / Decision:** EXPLAIN THAT THE WALL-THICKNESS DECODER GUARANTEES POSITIVITY BUT NOT ANATOMICAL ACCURACY. BUT THINK ABOUT IT


---

## 36. Explain the role of the SSM pretraining more carefully

The synthetic SSM data should be described as providing:
- geometric prior
- shape regularisation
- diverse plausible LV shapes
- controlled pretraining examples

It should not be described as reproducing real patient-specific wall thickness.

**Action / Decision:** EXPLAIN THAT THE SSM PRETRAINING PROVIDES GEOMETRIC PRIOR, SHAPE REGULARISATION, DIVERSE PLAUSIBLE LV SHAPES, AND CONTROLLED PRETRAINING EXAMPLES, BUT DOES NOT REPRODUCE REAL PATIENT-SPECIFIC WALL THICKNESS.


---

## 37. Clarify the role of phase conditioning

The model uses ED/ES phase conditioning.

Explain precisely what the phase variable contributes and why a single model is preferred over independent ED/ES models.

**Action / Decision:** EXPLAIN THE ROLE OF THE PHASE VARIABLE AND WHY A SINGLE MODEL IS PREFERRED OVER INDEPENDENT ED/ES MODELS. (I CHOSE IT BEACAUSE SSM ONLY HAD ED PHASE, SO TO BE ABLE TO USE THE SSM PRETRAINING, I HAD TO USE A SINGLE MODEL FOR BOTH PHASES.)


---

## 38. Explain the ES limitation

If the model receives a binary phase label but does not receive continuous cardiac phase information, acknowledge that the representation is relatively coarse.

Explain how this may contribute to ED/ES differences.

**Action / Decision:** EXPLAIN THAT THE MODEL RECEIVES A BINARY PHASE LABEL AND NOT CONTINUOUS CARDIAC PHASE INFORMATION, WHICH MAY CONTRIBUTE TO ED/ES DIFFERENCES. USE SIMPLE LANGUAGE TO EXPLAIN THIS LIMITATION.


---

# 🟣 RESULTS / STATISTICS

## 39. Re-run the Results chapter from the actual experiment outputs

**Problem:** There are enough contradictions that manually correcting individual numbers is risky.

Recommended procedure:
1. Identify the exact final checkpoint.
2. Identify the exact test set.
3. Re-run evaluation.
4. Generate all metrics directly from the same evaluation run.
5. Regenerate tables.
6. Rewrite prose from those tables.

**Action / Decision:** SEE THE NEW RESULTS WITH THE NEW ARCHITECTURE AND UPDATE THE TABLES AND PROSE TO MATCH.


---

## 40. Make tables the single source of truth

For every result:
- calculate it once
- put it in the table
- derive the surrounding prose directly from the table

Do not manually type the same numerical result in multiple places.

**Action / Decision:** MAKE TABLES THE SINGLE SOURCE OF TRUTH. UPDATE ALL PROSE TO MATCH THE TABLES.


---

## 41. Check all percentage calculations

Recalculate every:
- percentage difference
- volume ratio
- wall-thickness difference
- ED/ES percentage change
- improvement/degradation percentage

Use the exact same denominator throughout.

**Action / Decision:** CHECK


---

## 42. Check all ICC calculations

Verify:
- ICC model/type
- unit of analysis
- confidence intervals
- interpretation
- whether measurements are paired
- whether segments are treated as independent

**Action / Decision:** CHECK


---

## 43. Check Bland–Altman calculations

Verify:
- bias
- standard deviation of differences
- 95% limits of agreement
- units
- direction of subtraction

Make sure the reported values in the text exactly match the figure/table.

**Action / Decision:** CHECK


---

## 44. Check correlation calculations

Verify:
- Pearson vs. Spearman
- whether patient/segment dependence is relevant
- sample size
- confidence intervals if reported
- whether correlation is being confused with agreement

**Action / Decision:** CHECK


---

## 45. Do not equate high correlation with good agreement

A high Pearson \(r\) does not mean that two methods agree.

The thesis should distinguish:
- correlation
- bias
- limits of agreement
- ICC

**Action / Decision:** CHECK


---

# 🟤 WRITING / PRESENTATION

## 46. Remove duplicated Table of Contents entries

The current TOC contains duplicated front-matter items.

Check:
- Acknowledgments
- Resumo
- Abstract
- References

**Action / Decision:** LEAVE IT FOR NOW


---

## 47. Standardise terminology

Choose one consistent terminology for:
- endocardium / endocardial
- epicardium / epicardial
- wall thickness / myocardial thickness
- segmentation-derived reference
- model prediction
- reconstruction
- ground truth

**Action / Decision:** DO THIS.


---

## 48. Standardise ED/ES terminology

Define once:

- ED = end-diastole
- ES = end-systole

Then use ED/ES consistently throughout.

**Action / Decision:** DO THIS.

---

## 49. Avoid unsupported clinical claims

Be careful with statements implying:
- diagnostic utility
- clinical deployment
- clinical accuracy
- clinical decision support

unless directly demonstrated.

The thesis is primarily a methodological/reconstruction study.

**Action / Decision:** JUST TALK ABOUT WHAT IT IS. LEAVE THIS FOR THE CONCLUSION IN FUTURE WORK.


---

## 50. Make the limitations section genuinely honest

At minimum discuss:
- no independent 3D ground truth
- small test cohort
- limited pathology evaluation
- synthetic wall-thickness construction
- ES performance
- topology/post-processing
- repeated-measures statistics
- potential domain shift between datasets
- dependence on segmentation quality

**Action / Decision:** BE HONEST ABOUT LIMITATIONS.


---

# 🟢 STRONG PARTS TO PRESERVE

## 51. Preserve the explicit acknowledgement that 3D ground truth is unavailable

This is scientifically honest and should remain.

**Action / Decision:** PRESERVE.


---

## 52. Preserve the explanation of the synthetic-data limitation

The thesis correctly states that the synthetic epicardium is not intended to reproduce clinical wall thickness.

**Action / Decision:** PRESERVE.


---

## 53. Preserve the ED vs. ES analysis

The ES degradation is an interesting scientific finding and can strengthen the Discussion if presented honestly.

**Action / Decision:** PRESERVE.


---

## 54. Preserve the distinction between direct model thickness and independent thickness estimators

The thesis makes an important distinction:

- model-predicted thickness from \(\delta\)
- independent thickness algorithms applied to reconstructed geometry

This is conceptually useful.

**Action / Decision:**\ PRESERVE.


---

## 55. Preserve the multi-method thickness evaluation

Comparing:
- Laplace
- Yezzi–Prince
- SDF cone rays
- EDT

gives the thesis a useful methodological comparison.

**Action / Decision:** PRESERVE.


---

# Final assessment

The thesis has a coherent and potentially good research contribution, but the current PDF contains enough contradictions that it should **not be submitted in its current state**.

The main task is not to invent more experiments or add more literature.

The main task is to make the existing work **internally consistent, reproducible, correctly referenced, and scientifically precise**.

