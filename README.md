# Policy-Aware Crisis Report Triage

This repository contains the final MSAI 699 capstone code for an explainable NLP system that triages open-source crisis messages for human review.

## Project purpose

The model predicts whether a crisis-related social-media message is informative enough to elevate to a human analyst. It is designed for decision support, not autonomous emergency response.

## Dataset

The working file is `data/crisislex_t26_baseline_subset.csv`, containing 5,149 unique CrisisLexT26 messages from five crisis events. The positive class is `Related and informative`; all other labels are mapped to `Not informative`.

## Main models

- **Baseline:** TF-IDF unigrams/bigrams + logistic regression with balanced class weights.
- **Hybrid:** normalized word TF-IDF + character n-grams + logistic regression.
- **Reliability policy:** escalate to review if either fixed model predicts the message as informative.

## Reproduce the final tests

```bash
pip install -r requirements.txt
python src/final_testing.py
```

This writes cross-event results to the `results/` folder. The notebook in `notebooks/Final_Capstone_Crisis_Triage.ipynb` provides a readable walkthrough of the same methodology.

## Run the demo predictor

```bash
python src/predict.py --text "Emergency officials report mandatory evacuation downtown; shelter open at Central High School."
```

## Run the optional FastAPI service

```bash
uvicorn src.app:app --reload
```

Then send a POST request to `/predict` with JSON such as:

```json
{"text": "Emergency officials report mandatory evacuation downtown; shelter open at Central High School."}
```

## Final recommendation

Random-test performance was strong, but leave-one-event-out testing showed weaker generalization to unseen crises. The system should therefore route positive and borderline messages to human review, monitor event-level performance, and never autonomously discard crisis reports.
