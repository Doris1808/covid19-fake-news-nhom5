"""EDA: phân bố nhãn, độ dài, trùng văn bản, ví dụ có truy xuất từ train."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .common import ROOT, OUT, load_split, save_json, normalize_for_overlap


def run(data_dir=None, out_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data_dir = Path(data_dir) if data_dir else ROOT/'data'/'raw'
    out_dir = Path(out_dir) if out_dir else OUT/'eda'
    out_dir.mkdir(parents=True,exist_ok=True)
    frames={s:load_split(s,data_dir) for s in ['train','val','test']}
    rows=[]; examples=[]; overlap={}
    for split,df in frames.items():
        lens=df.tweet.str.split().str.len()
        rows.append({'split':split,'count':int(len(df)),'real':int((df.label=='real').sum()),
                     'fake':int((df.label=='fake').sum()),'avg_words':round(float(lens.mean()),2),
                     'median_words':float(lens.median()),'p95_words':float(lens.quantile(.95)),
                     'min_words':int(lens.min()),'max_words':int(lens.max())})
        if split=='train':
            for label in ['real','fake']:
                examples.extend(df[df.label==label].head(2).assign(split=split)[['split','id','tweet','label']].to_dict('records'))
    for left,right in [('train','val'),('train','test'),('val','test')]:
        a={normalize_for_overlap(t) for t in frames[left].tweet}
        b={normalize_for_overlap(t) for t in frames[right].tweet}
        overlap[f'{left}_{right}']=len(a & b)
    summary=pd.DataFrame(rows)
    summary.to_csv(out_dir/'dataset_stats.csv',index=False)
    pd.DataFrame(examples).to_csv(out_dir/'examples_train.csv',index=False)
    save_json({'splits':rows,'exact_normalized_overlap_count':overlap},out_dir/'eda.json')
    summary.set_index('split')[['real','fake']].plot(kind='bar',rot=0)
    plt.title('Phân bố nhãn real / fake (dữ liệu thật)')
    plt.xlabel('Tập dữ liệu');plt.ylabel('Số mẫu');plt.tight_layout()
    plt.savefig(out_dir/'label_distribution.png',dpi=160);plt.close()
    text=['# Báo cáo EDA (đo trực tiếp từ CSV)', '', summary.to_markdown(index=False), '',
          '## Trùng văn bản giữa các split (chuẩn hoá chữ thường và khoảng trắng)', '']
    text += [f'- {k}: {v} văn bản' for k,v in overlap.items()]
    text += ['', '## Ví dụ (lấy 2 mẫu đầu tiên mỗi nhãn của train)', '']
    for ex in examples:
        text.append(f'- **{ex["label"]}**, id={ex["id"]}: {ex["tweet"][:350]}')
    text += ['', '> Lưu ý: ví dụ có nhãn phản ánh dataset, không chứng minh tính đúng/sai ngoài thực tế.']
    (out_dir/'eda.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(summary.to_string(index=False))
    print('Đã lưu',out_dir)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir');p.add_argument('--out-dir')
    a=p.parse_args();run(a.data_dir,a.out_dir)
