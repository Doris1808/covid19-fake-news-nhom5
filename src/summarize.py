"""Tổng hợp Validation từ các lượt DistilBERT đã hoàn thành."""
import json
from pathlib import Path
import pandas as pd
from .common import ROOT


def collect(root):
    rows = []
    signatures = []
    for path in sorted(Path(root).glob('seed_*/metrics.json')):
        m = json.loads(path.read_text())
        if m.get('smoke') or not (path.parent/'completed.flag').exists():
            continue
        cfg = json.loads((path.parent/'run_config.json').read_text())
        signatures.append(json.dumps({k:v for k,v in cfg.items() if k != 'seed'}, sort_keys=True))
        rows.append(dict(seed=m['seed'], **{k:v for k,v in m['val'].items() if isinstance(v, float)},
                         train_seconds_this_session=m['train_seconds_this_session'],
                         resumed=m['resumed'], peak_cuda_allocated_gib=m['peak_cuda_allocated_gib']))
    frame = pd.DataFrame(rows)
    output = Path(root); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'summary.csv', index=False)
    lines = ['# Kết quả DistilBERT trên Validation', '', f'Số seed hoàn thành: {len(rows)}/3.', '']
    if len(rows):
        lines.append(frame.to_markdown(index=False))
    if len(rows) == 3 and set(frame.seed) == {42,123,2024} and len(set(signatures)) == 1:
        lines += ['', f"F1 weighted: {frame.f1_weighted.mean():.4f} ± {frame.f1_weighted.std(ddof=1):.4f} (độ lệch chuẩn mẫu)."]
    else:
        lines += ['', 'Chưa tính trung bình 3 seed: cần đủ seed, cùng cấu hình và cùng dữ liệu.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    return frame

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--root', default=str(ROOT/'outputs/distilbert'))
    collect(p.parse_args().root)
