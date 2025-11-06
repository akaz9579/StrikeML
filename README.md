# **StrikeML**

**StrikeML** is a data-driven project that predicts whether a *motion to strike* in Connecticut civil court cases will be **granted** or **denied**.

It uses real court data to train and test a machine learning model on patterns found in attorney information, case types, and motion details.

---

## **Project Purpose**

In civil litigation, lawyers often file motions to strike early in a case — asking the court to remove part or all of the opponent’s claims.

The goal of StrikeML is to explore whether these motion outcomes can be predicted **using only the data available during the case itself**, not after the trial.

This helps simulate real-time legal decision support and builds on research showing how data and AI can assist fairer, more consistent legal outcomes.

---

## **Data Overview**

The dataset is drawn from **Connecticut Judicial Branch civil case records (2004–2019)**.

It contains both structured court data and motion-level information.

**Key Features Used:**

- **Attorney ID (Juris Number):** Unique identifier for each lawyer or firm
- **Attorney Specialization:** Calculated using an entropy formula to measure how focused an attorney’s practice is
- **Case Type:** Tort (injury) or Vehicular (car accident)
- **Case Location:** Which Connecticut court the case is in
- **Motion Metadata:** Filing party, duration, and document type
- **Outcome:** Whether the motion to strike was granted or denied

---

## **Model Goal**

This is a **binary classification problem** — predicting if a motion will be *granted* (1) or *denied* (0).

The model is trained on part of the dataset and tested on unseen data to measure how accurately it can predict outcomes.

**Evaluation Metrics:**

- Accuracy
- True/False Positives and Negatives
- Precision, Recall, and F1 Score

---

## **Method Summary**

StrikeML uses machine learning models such as:

- Decision Trees
- Random Forests
- AdaBoost
- Gradient Boosting
- XGBoost
- Support Vector Machines

The model is trained on structured CSV data, with options to expand into natural language features (like complaint document text) for future analysis.

---

## **Results (from reference study)**

As we do not continue foward with the word emedding for this specific instance of the project, we mock similar results to the inital paper, ranging from 55-58% accuracy across models.
