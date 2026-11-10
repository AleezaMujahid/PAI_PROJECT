# Fairness-Aware Credit Default Prediction System

**Machine Learning Semester Project \| German Credit & Taiwan Credit
Card Default**

A local, laptop-runnable machine learning application that predicts
credit default risk and audits potential differences in model outcomes
across demographic groups.

> **Research disclaimer:** A prediction of `Default (1)` is an estimated
> credit-default risk classification, **not** an actual loan rejection
> or proof of discriminatory lending. This project is for education and
> research, not real-world lending decisions.

## 1. Project Name

**Fairness-Aware Credit Default Prediction Using Machine Learning**

## 2. Objective

-   Predict whether a borrower is likely to default (`1`) or not default
    (`0`).
-   Compare Logistic Regression and Random Forest classifiers.
-   Improve predictive performance through development-only
    hyperparameter and classification-threshold tuning.
-   Evaluate accuracy, precision, recall, F1-score, ROC-AUC, and average
    precision.
-   Audit model outcomes across age and sex groups, and foreign-worker
    status where available.
-   Investigate fairness mitigation approaches and their performance
    trade-offs.
-   Provide a **Streamlit web interface running locally on a laptop** to
    demonstrate predictions from previously saved models.

## 3. Project Description

The project analyzes two credit-default datasets. It performs data
cleaning, stratified splitting, preprocessing, cross-validation, model
training, hyperparameter tuning, threshold selection, fairness analysis,
and final evaluation. The final trained models are saved and used by a
Streamlit application for interactive predictions on saved test
examples.

The system has two parts:

1.  **Offline machine learning pipeline:** preprocesses data, trains and
    compares models, evaluates fairness, and saves models and reports.
2.  **Live demonstration frontend:** loads the saved models and allows a
    user to select a dataset and an existing test customer to view the
    prediction.

**The frontend does not need to retrain the models.**

## 4. Datasets

  --------------------------------------------------------------------------
  Dataset   Cleaned rows        Train   Validation     Original      Encoded
                                                           test   predictors
  --------- ------------ ------------ ------------ ------------ ------------
  German           1,000          700          149          151           54
  Credit                                                        

  Taiwan          29,965       20,975        4,494        4,496           23
  Credit                                                        
  Card                                                          
  Default                                                       
  --------------------------------------------------------------------------

-   **Target:** `0` = No Default, `1` = Default.
-   **Original split membership:** preserved using the saved `row_id`
    values.
-   **Sensitive attributes:** stored separately for fairness auditing
    rather than directly used as model predictors.

## 5. Algorithms and Methods

### Logistic Regression

A binary classification model that estimates the probability of default.
It is interpretable and provides a useful baseline for credit-risk
modeling. The final German model uses Logistic Regression.

### Random Forest

An ensemble of decision trees that can capture nonlinear patterns and
interactions. The final Taiwan model uses Random Forest.

### Supporting methods

-   Stratified training, validation, and test splits.
-   Missing-value imputation, categorical one-hot encoding, and numeric
    scaling where applicable.
-   Five-fold cross-validation and out-of-fold development predictions.
-   Hyperparameter tuning and decision-threshold tuning **on development
    data only**.
-   Confusion-matrix and probability-based evaluation.
-   Group fairness metrics, bootstrap confidence intervals, and
    exploratory mitigation experiments.
-   Fairness reweighing and group-specific threshold mitigation studied
    during development; these mitigations are **not silently applied**
    to the selected final models.

## 6. Mathematical Foundations

  -------------------------------------------------------------------------
  Area                    Concept                   Where it supports the
                                                    project
  ----------------------- ------------------------- -----------------------
  Linear algebra          Feature vectors and       Logistic Regression
                          weighted sums             scoring

  Calculus                Loss functions,           Training Logistic
                          gradients, optimization   Regression

  Statistics              Mean, standard deviation, Preprocessing,
                          stratified sampling, CV,  validation, uncertainty
                          bootstrap                 

  Probability             Conditional default       Predicting `Default`
                          probabilities and         versus `No Default`
                          thresholds                

  Classification metrics  Confusion matrix,         Evaluating model
                          precision, recall, F1     performance

  Fairness statistics     Predicted-default-rate,   Comparing demographic
                          FPR, and FNR gaps         groups
  -------------------------------------------------------------------------

Key equations:

**Logistic Regression score and probability**

\[
z=`\mathbf{w}`{=tex}\^{T}`\mathbf{x}`{=tex}+b,`\qquad `{=tex}P(Y=1`\mid `{=tex}X)=`\frac{1}{1+e^{-z}}`{=tex}
\]

**Binary cross-entropy**

\[
L=-`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}`\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]`{=tex}\]

**Precision, recall, and F1**

\[ `\mathrm{Precision}`{=tex}=`\frac{TP}{TP+FP}`{=tex},`\quad`{=tex}
`\mathrm{Recall}`{=tex}=`\frac{TP}{TP+FN}`{=tex},`\quad`{=tex}
F_1=`\frac{2PR}{P+R}`{=tex} \]

**Predicted-default-rate gap**

\[ `\Delta`{=tex}\_{`\mathrm{PDR}`{=tex}}=
P(`\hat{Y}`{=tex}=1`\mid `{=tex}A=`\text{under 25}`{=tex})-
P(`\hat{Y}`{=tex}=1`\mid `{=tex}A=`\text{25+}`{=tex}) \]

The mathematical chapter in the project report provides further
explanation and application details.

## 7. Final Selected Models

These choices were locked before the corrected original-split final
evaluation.

  -----------------------------------------------------------------------------------
  Dataset          Model            Selected hyperparameters       Decision threshold
  ---------------- ---------------- --------------------------- ---------------------
  German           Logistic         `C=0.1`,                                 **0.40**
                   Regression       `class_weight="balanced"`   

  Taiwan           Random Forest    `n_estimators=300`,                      **0.55**
                                    `min_samples_leaf=10`,      
                                    `class_weight="balanced"`   
  -----------------------------------------------------------------------------------

A predicted default probability **at or above** the threshold produces
`Default (1)`.

## 8. Development Performance (After Tuning)

Development results below are based on out-of-fold evaluation and
selected thresholds.

  -----------------------------------------------------------------------------
  Dataset     Accuracy   Precision     Recall         F1    ROC-AUC     Average
                                                                      precision
  --------- ---------- ----------- ---------- ---------- ---------- -----------
  German        0.6572      0.4610     0.8353     0.5941     0.7742      0.5877

  Taiwan        0.7949      0.5346     0.5634     0.5486     0.7797      0.5528
  -----------------------------------------------------------------------------

**Before-tuning note:** Baseline models were evaluated during
development, but a complete verified set of their numeric metrics is not
reproduced here. Consult the original model-comparison reports for exact
before-versus-after figures; do not substitute the final test values for
baseline values.

## 9. Corrected Original-Split Final Test Results

  Metric                       German       Taiwan
  ---------------------- ------------ ------------
  Test rows                       151        4,496
  Accuracy                 **0.6755**   **0.7903**
  Precision                    0.4730       0.5248
  Recall                       0.7778       0.5538
  F1-score                 **0.5882**   **0.5389**
  ROC-AUC                  **0.8057**   **0.7802**
  Average precision            0.6749       0.5606
  True negatives (TN)              67        3,002
  False positives (FP)             39          499
  False negatives (FN)             10          444
  True positives (TP)              35          551

The datasets differ in size and default prevalence, so raw accuracy is
not sufficient for declaring one model superior to the other.

### Evaluation protocol note

An earlier provisional evaluation accidentally reconstructed
validation/test membership, producing different test counts. The
evaluation was corrected using the **original saved split IDs** and the
**original train-fitted processed features**. The selected models,
hyperparameters, and thresholds remained unchanged. Because provisional
test outcomes had already been viewed, the test set should not be
described as literally never inspected.

## 10. Final Test Fairness Findings

  -----------------------------------------------------------------------
  Dataset /         Comparison        Predicted default False positive
  attribute                           rate              rate
  ----------------- ----------------- ----------------- -----------------
  German / age      Under 25 vs 25+   57.14% vs 47.15%  46.67% vs 35.16%

  German / sex      Female vs male    48.15% vs 49.48%  31.43% vs 39.44%

  Taiwan / age      Under 25 vs 25+   32.33% vs 22.48%  21.14% vs 13.61%

  Taiwan / sex      Female vs male    21.50% vs 26.20%  12.74% vs 16.72%
  -----------------------------------------------------------------------

For German foreign-worker status, the `no` subgroup contained only **5
test observations**; its estimates are highly unstable and should be
treated as exploratory.

**Interpretation:** Younger borrowers had higher predicted-default rates
and higher false-positive rates in both datasets. These differences
warrant further investigation but do not alone prove unfair treatment or
establish causation.

## 11. Project Folder Structure

``` text
PAI_PROJECT/
├── app.py                            # Streamlit frontend
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── cv.py
│   ├── fairness_compare.py
│   ├── fairness_bootstrap.py
│   ├── fairness_mitigation.py
│   ├── threshold_fairness_mitigation.py
│   ├── compare_models.py
│   ├── final_model_selection.py
│   └── final_test_evaluation_fixed.py
├── data/
│   ├── raw/                          # Source datasets (local)
│   └── processed/
│       ├── german/
│       │   ├── X_test.csv
│       │   ├── y_test.csv
│       │   └── ...
│       └── taiwan/
│           ├── X_test.csv
│           ├── y_test.csv
│           └── ...
├── reports/
│   ├── final_model_selection/
│   ├── hyperparameter_tuning/
│   └── final_test_evaluation_original_splits/
│       ├── german/
│       │   ├── model.joblib
│       │   ├── metrics.json
│       │   ├── predictions.csv
│       │   ├── fairness_age_group.csv
│       │   ├── fairness_sex.csv
│       │   ├── fairness_foreign_worker.csv
│       │   ├── confusion_matrix.png
│       │   ├── roc_curve.png
│       │   └── precision_recall_curve.png
│       └── taiwan/
│           ├── model.joblib
│           ├── metrics.json
│           ├── predictions.csv
│           ├── fairness_age_group.csv
│           ├── fairness_sex.csv
│           ├── confusion_matrix.png
│           ├── roc_curve.png
│           └── precision_recall_curve.png
└── .venv/                            # Local Python virtual environment
```

The structure above describes the main files used in the project; your
local folder may contain additional scripts and reports.

## 12. How to Run the Project on a Laptop

### Prerequisites

-   Windows laptop with Python and VS Code.
-   Project folder `PAI_PROJECT`.
-   Saved model and processed test data files listed above.
-   Python packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`,
    `joblib`, `matplotlib`.

### Step 1 --- Open the project

Open the `PAI_PROJECT` folder in VS Code. In the PowerShell terminal:

``` powershell
cd C:\Users\dell\PAI_PROJECT
```

### Step 2 --- Activate the existing virtual environment

``` powershell
.\.venv\Scripts\Activate.ps1
```

If you do not already have `.venv`, create it first:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 --- Install required packages

``` powershell
python -m pip install streamlit pandas numpy scikit-learn joblib matplotlib
```

Use your project's existing dependency lockfile, if available, to
reproduce the original environment more precisely.

### Step 4 --- Start the Streamlit frontend

``` powershell
python -m streamlit run app.py
```

Open **http://localhost:8501** in a browser if it does not open
automatically.

### Step 5 --- Demonstrate a prediction

1.  Choose **German Credit** or **Taiwan Credit**.
2.  Select a customer from the **original test dataset**.
3.  Click **Predict default risk**.
4.  Observe the model's predicted class, estimated default probability,
    and threshold.
5.  Reveal or compare the **actual label** for educational evaluation.
6.  Explore the saved performance metrics, confusion matrix, and
    fairness plots.

**Important:** The model receives only customer feature columns from
`X_test.csv`. The `y_test.csv` target is **not an input feature**; it is
used solely to compare the predicted outcome against the known outcome.

### What is actually running?

``` text
Streamlit frontend
       |
       v
Load saved test features (X_test.csv)
       |
       v
Load saved model (model.joblib)
       |
       v
model.predict_proba(customer_features)
       |
       v
Apply fixed threshold (German 0.40 / Taiwan 0.55)
       |
       v
Show Default (1) or No Default (0)
       |
       v
Optionally compare with actual label (y_test.csv)
```

The frontend performs **inference**, not model retraining or
hyperparameter tuning.

## 13. Running the Research Pipeline (Optional)

The full research pipeline is separate from the live demo. Existing
scripts support preprocessing, training, cross-validation, fairness
comparisons, tuning, and report generation.

**Do not rerun preprocessing or final evaluation casually.** The
original saved splits and final results should remain intact. If
reproducing the entire research experiment, work on a separate copy of
the project and preserve original split membership.

## 14. Troubleshooting

### `StreamlitDuplicateElementId`

This means identical Streamlit widgets were created without unique
identifiers, or the application code was duplicated. Give each widget a
unique `key`, **and remove duplicate widget blocks** if the same
interface was pasted twice.

Example:

``` python
choice = st.sidebar.selectbox(
    "Choose dataset",
    ["german", "taiwan"],
    key="dataset_selector",
)
index = st.number_input(
    "Test customer number (1-based)",
    min_value=1,
    max_value=len(X),
    value=1,
    step=1,
    key="test_customer_number",
)
```

Restart Streamlit after saving changes:

``` powershell
# Stop the current app with Ctrl+C, then:
python -m streamlit run app.py
```

### `FileNotFoundError`

Verify that `app.py`, the saved `model.joblib`, `X_test.csv`,
`y_test.csv`, and report folders are present at the expected paths.

### `ModuleNotFoundError`

Activate `.venv` and install the missing package:

``` powershell
python -m pip install streamlit pandas numpy scikit-learn joblib matplotlib
```

### Feature mismatch

Use the **saved original processed test features** and the corresponding
saved model. Do not reorder feature columns, insert `row_id` as a
predictor, or apply a different preprocessing pipeline.

### `FileExistsError` from final evaluation

This may be a deliberate protection against overwriting existing final
test reports. **Do not delete saved final reports just to bypass this
safeguard.** The frontend can run directly from the existing saved
model.

## 15. Challenges, Solutions, and Limitations

  -----------------------------------------------------------------------
  Challenge                           Approach
  ----------------------------------- -----------------------------------
  Data leakage                        Fit preprocessing on
                                      development/training data, never on
                                      held-out test data

  Class imbalance                     Use balanced class weights and
                                      evaluate recall, F1, and
                                      precision--recall performance

  Model selection                     Use development cross-validation,
                                      tuning, and locked thresholds

  Split reconstruction mismatch       Restore exact membership using
                                      original saved `row_id` values

  Circular/self-import error          Keep preprocessing implementation
                                      separate from evaluation imports

  Streamlit duplicate widgets         Use unique widget keys and
                                      eliminate duplicate UI blocks

  Small demographic subgroups         Report sample sizes and treat
                                      unstable fairness results
                                      cautiously

  Fairness--performance trade-off     Study mitigation on development
                                      data and report both benefits and
                                      costs
  -----------------------------------------------------------------------

Additional limitations:

-   German Credit has a relatively small test sample.
-   Fairness disparities may reflect differences in base rates, sample
    composition, or unmeasured factors; observational metrics do not
    establish causality.
-   Sensitive attributes are excluded from direct predictors, but
    correlated variables may still act as proxies.
-   Group-specific threshold mitigation is a research simulation and not
    automatically suitable for deployment.
-   The test-customer frontend demonstrates predictions on already
    processed examples; **manual raw-customer entry would require the
    exact saved training-time preprocessing pipeline**.
-   Saved model files should be loaded only from trusted project
    sources; serialized Python model files are not safe to open from
    unknown origins.
-   The project is not a production-ready or legally validated lending
    decision system.

## 16. Conclusion

The project demonstrates an end-to-end workflow for credit-default
prediction and group fairness auditing. The selected German Logistic
Regression model achieved **67.55% accuracy**, **0.5882 F1**, and
**0.8057 ROC-AUC** on the corrected original test split. The selected
Taiwan Random Forest model achieved **79.03% accuracy**, **0.5389 F1**,
and **0.7802 ROC-AUC**.

The fairness audit found differences in predicted-default and
false-positive rates across demographic groups, especially by age. These
findings show why predictive accuracy and fairness must be assessed
separately. The local Streamlit application provides an interactive
demonstration of the saved models without retraining them.

------------------------------------------------------------------------

**Project status:** Model development, final original-split evaluation,
and fairness auditing completed. Streamlit frontend provided for laptop
demonstration; verify that the local interface runs without
duplicate-widget errors before presenting.
