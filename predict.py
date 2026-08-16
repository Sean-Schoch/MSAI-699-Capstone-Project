import argparse
import json
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

from modeling import load_data, make_baseline_model, make_hybrid_model

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'crisislex_t26_baseline_subset.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

BASELINE_PATH = MODELS / 'baseline_model.joblib'
HYBRID_PATH = MODELS / 'hybrid_model.joblib'

def load_or_train():
    if BASELINE_PATH.exists() and HYBRID_PATH.exists():
        return joblib.load(BASELINE_PATH), joblib.load(HYBRID_PATH)
    df = load_data(DATA_PATH)
    X = df['tweet_text']
    y = df['binary_label'].astype(int)
    baseline = make_baseline_model().fit(X, y)
    hybrid = make_hybrid_model().fit(X, y)
    joblib.dump(baseline, BASELINE_PATH)
    joblib.dump(hybrid, HYBRID_PATH)
    return baseline, hybrid

def predict(text: str):
    baseline, hybrid = load_or_train()
    baseline_prob = float(baseline.predict_proba([text])[0, 1])
    hybrid_prob = float(hybrid.predict_proba([text])[0, 1])
    baseline_flag = baseline_prob >= 0.50
    hybrid_flag = hybrid_prob >= 0.47
    review = baseline_flag or hybrid_flag
    return {
        'text': text,
        'baseline_probability': baseline_prob,
        'hybrid_probability': hybrid_prob,
        'baseline_threshold': 0.50,
        'hybrid_threshold': 0.47,
        'review_recommendation': 'Elevate for human review' if review else 'Lower priority; do not discard without policy review',
        'policy': 'Escalate if either baseline or hybrid model is positive',
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.text), indent=2))
