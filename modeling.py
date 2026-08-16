
import html
import re
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline

RANDOM_STATE = 42
BASELINE_THRESHOLD = 0.50
HYBRID_THRESHOLD = 0.47


def normalize_text(text):
    """Text normalization retained from the optimization phase."""
    text = html.unescape(str(text))
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'@\w+', ' USER ', text)
    text = re.sub(r'#([A-Za-z0-9_]+)', r' \1 ', text)
    text = re.sub(r'\bRT\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {'tweet_text', 'original_label', 'event', 'binary_label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df['tweet_text'].isna().any():
        raise ValueError('Missing tweet_text values detected.')
    if not set(df['binary_label'].unique()) <= {0, 1}:
        raise ValueError('binary_label must be binary.')
    if df['event'].nunique() != 5:
        raise ValueError('This experiment expects the five-event baseline subset.')
    return df.reset_index(drop=True)


def make_baseline_model() -> Pipeline:
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=20000,
            sublinear_tf=True,
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
        )),
    ])


def make_hybrid_model() -> Pipeline:
    return Pipeline([
        ('features', FeatureUnion([
            ('word', TfidfVectorizer(
                lowercase=True,
                strip_accents='unicode',
                preprocessor=normalize_text,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                max_features=15000,
                sublinear_tf=True,
            )),
            ('char', TfidfVectorizer(
                lowercase=True,
                preprocessor=normalize_text,
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=2,
                max_features=30000,
                sublinear_tf=True,
            )),
        ])),
        ('classifier', LogisticRegression(
            max_iter=1500,
            C=2.0,
            class_weight=None,
            random_state=RANDOM_STATE,
        )),
    ])


def metric_dict(y_true, y_pred, y_prob) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
    }


def leave_one_event_out(df: pd.DataFrame) -> pd.DataFrame:
    models = {
        'Baseline 0.50': (make_baseline_model(), BASELINE_THRESHOLD),
        'Hybrid 0.47': (make_hybrid_model(), HYBRID_THRESHOLD),
    }
    per_event_rows = []
    oof_rows = []
    for held_event in df['event'].drop_duplicates():
        train = df[df['event'] != held_event].copy()
        test = df[df['event'] == held_event].copy()
        for model_name, (model_template, threshold) in models.items():
            model = clone(model_template)
            model.fit(train['tweet_text'], train['binary_label'])
            prob = model.predict_proba(test['tweet_text'])[:, 1]
            pred = (prob >= threshold).astype(int)
            metrics = metric_dict(test['binary_label'], pred, prob)
            per_event_rows.append({
                'Held-out event': held_event,
                'Model': model_name,
                'N': len(test),
                'Positive rate': float(test['binary_label'].mean()),
                **metrics,
            })
            fold = test[['tweet_text', 'original_label', 'event', 'binary_label']].copy()
            fold['Model'] = model_name
            fold['Threshold'] = threshold
            fold['Probability'] = prob
            fold['Prediction'] = pred
            oof_rows.append(fold)
    per_event = pd.DataFrame(per_event_rows)
    oof = pd.concat(oof_rows, ignore_index=True)
    base = oof[oof['Model'] == 'Baseline 0.50'].reset_index(drop=True)
    hybrid = oof[oof['Model'] == 'Hybrid 0.47'].reset_index(drop=True)
    if not base['tweet_text'].equals(hybrid['tweet_text']):
        raise ValueError('Model folds are not aligned.')
    union_pred = ((base['Prediction'].to_numpy() == 1) | (hybrid['Prediction'].to_numpy() == 1)).astype(int)
    union_prob = np.maximum(base['Probability'].to_numpy(), hybrid['Probability'].to_numpy())
    union = base[['tweet_text', 'original_label', 'event', 'binary_label']].copy()
    union['Model'] = 'Escalate if either positive'
    union['Probability'] = union_prob
    union['Prediction'] = union_pred
    for held_event, test in union.groupby('event', sort=False):
        metrics = metric_dict(test['binary_label'], test['Prediction'], test['Probability'])
        per_event_rows.append({
            'Held-out event': held_event,
            'Model': 'Escalate if either positive',
            'N': len(test),
            'Positive rate': float(test['binary_label'].mean()),
            **metrics,
        })
    return pd.DataFrame(per_event_rows)


def pooled_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in results.groupby('Model', sort=False):
        tn, fp, fn, tp = g[['TN','FP','FN','TP']].sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        rows.append({
            'Model': model,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'False negatives': int(fn),
            'False positives': int(fp),
            'Review rate': (tp+fp)/(tp+tn+fp+fn),
        })
    return pd.DataFrame(rows)
