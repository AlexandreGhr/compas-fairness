# Fairness in Classification on the COMPAS Dataset

Can a recidivism prediction algorithm be both accurate and fair? This project analyzes the racial bias of **COMPAS**, a risk assessment tool used in US courts, then trains and compares machine learning classifiers and tests several techniques to make them fairer.

> Data Science project — Master MoSIG (Université Grenoble Alpes), April 2026.
> Team of 4: **Alexandre Gauchier**, Elouann Marfil, Tom Laucournet, Alexander Ostle.

---

## Context

COMPAS predicts the risk that a defendant will reoffend, with a score from 1 to 10 based on 137 questions. In 2016, a ProPublica investigation showed that Black defendants received disproportionately high scores. We use ProPublica's dataset (~6,000 defendants from Broward County, Florida, 2013–2014) and the label `two_year_recid` (re-arrest within two years).

We evaluate fairness between Black and White defendants with three standard criteria:

| Criterion | Definition |
|---|---|
| **Independence** | Same rate of "at risk" predictions across groups |
| **Separation** *(main criterion)* | Same false positive and false negative rates (FPR, FNR) across groups |
| **Sufficiency** | Among people labeled "at risk", same proportion actually reoffends across groups |

These three criteria cannot be satisfied simultaneously when recidivism rates differ between groups (Chouldechova, 2017).

---

## Project structure

The whole analysis is in [`compas_fairness.ipynb`](compas_fairness.ipynb):

1. **Analysis of COMPAS** — score distributions, calibration and error rates by race.
2. **Standard classifiers** — 7 classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes) and 10 feature combinations, compared on performance and fairness.
3. **Fair classifier** — excluding race, resampling (under/oversampling) and post-processing with group-specific thresholds.

---

## Key results

**COMPAS is biased in terms of separation.** It is well calibrated for both groups (sufficiency), but Black defendants who did not reoffend are about twice as often labeled "at risk" (FPR 42.3% vs 22.0%), while White defendants who did reoffend are more often missed (FNR 49.6% vs 28.5%).

**Simple models match COMPAS.** With only 7 features (age, sex, criminal history, charge degree; race excluded), Gradient Boosting reaches 67.6% accuracy (AUC 0.720), slightly above COMPAS (~66%), with a smaller FPR gap (0.173 vs 0.203). Logistic Regression is a close and more interpretable alternative (AUC 0.717, FPR gap 0.143).

**Fairness techniques have a cost:**

| Approach (Gradient Boosting, 7 features) | FPR gap | FNR gap | PPV gap | Accuracy |
|---|---|---|---|---|
| Baseline | 0.173 | -0.214 | 0.054 | 67.6% |
| Undersampling | 0.179 | -0.228 | 0.069 | 68.5% |
| Oversampling | 0.150 | -0.173 | 0.063 | 66.5% |
| **Post-processing (group thresholds)** | **0.000** | **0.011** | 0.138 | 65.7% |

Post-processing equalizes the error rates almost perfectly, but lowers accuracy, increases the PPV gap (sufficiency is lost, as the theory predicts), and requires using race explicitly at decision time, which is ethically debatable.

---

## Limitations

- **Biased labels**: `two_year_recid` measures re-arrests, not actual offenses. Unequal policing affects the data itself, so every model trained on it learns this bias.
- **Proxies**: removing race does not remove the bias, because other features (such as prior counts) are correlated with race.
- **Resampling evaluation**: data is resampled before the train/test split, so resampling results are only indicative.
- **Data leakage, identified and fixed**: an earlier version of this project used `is_violent_recid`, which records violent reoffending *after* the COMPAS screening. This leaked information about the target and inflated the results (AUC 0.78). It was removed, and all results above are without it.

---

## How to run

```bash
git clone https://github.com/AlexandreGhr/compas-fairness.git
cd compas-fairness
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `compas_fairness.ipynb` in Jupyter or VS Code and run all cells. The dataset is downloaded automatically from the [ProPublica repository](https://github.com/propublica/compas-analysis) on first run.

---

## References

- J. Angwin et al., *Machine Bias*, ProPublica, 2016.
- A. Chouldechova, *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*, Big Data, 2017.
- M. Hardt, E. Price, N. Srebro, *Equality of Opportunity in Supervised Learning*, NeurIPS 2016.