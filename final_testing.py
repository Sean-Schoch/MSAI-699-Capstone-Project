
from pathlib import Path
import matplotlib.pyplot as plt

from modeling import load_data, leave_one_event_out, pooled_results

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'crisislex_t26_baseline_subset.csv'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

def main():
    df = load_data(DATA_PATH)
    results = leave_one_event_out(df)
    results.to_csv(RESULTS / 'reproduced_cross_event_results.csv', index=False)
    pooled = pooled_results(results)
    pooled.to_csv(RESULTS / 'reproduced_pooled_results.csv', index=False)
    print(pooled.to_string(index=False))
    plot_df = results.pivot(index='Held-out event', columns='Model', values='F1-score')
    ax = plot_df.plot(kind='bar', figsize=(11, 5))
    ax.set_ylabel('F1-score')
    ax.set_title('Leave-One-Event-Out F1-score')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTS / 'reproduced_cross_event_f1.png', dpi=180)

if __name__ == '__main__':
    main()
