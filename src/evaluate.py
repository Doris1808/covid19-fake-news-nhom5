"""Đánh giá Test riêng, sau khi chốt cấu hình bằng Validation."""
import argparse
import json
from pathlib import Path
from .common import ROOT, load_split, scores, save_json, fingerprint
from .predict import load_model, predict


def run(run_dir, data_dir):
    run_dir = Path(run_dir)
    config = json.loads((run_dir/'run_config.json').read_text())
    if config['smoke']:
        raise ValueError('Không đánh giá Test cho lượt smoke.')
    if not (run_dir/'completed.flag').exists():
        raise ValueError('Lượt train chưa hoàn tất.')
    if (run_dir/'test_metrics.json').exists():
        raise FileExistsError('Đã đánh giá Test cho lượt chạy này.')
    tokenizer, model = load_model(run_dir/'model')
    frame = load_split('test', Path(data_dir))
    preds = [max(p, key=p.get) for p in
             (predict(t, tokenizer, model, config['max_length']) for t in frame.tweet)]
    result = dict(scores(frame.label.tolist(), preds),
                  dataset_sha256=fingerprint(Path(data_dir)/'test.csv'))
    frame.assign(predicted_label=preds).to_csv(run_dir/'test_predictions.csv', index=False)
    save_json(result, run_dir/'test_metrics.json')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--data-dir', default=str(ROOT/'data/raw'))
    p.add_argument('--confirm-config-frozen', action='store_true', required=True)
    a = p.parse_args()
    run(a.run_dir, a.data_dir)
