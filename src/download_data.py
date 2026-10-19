"""Tải đúng 3 file CSV công khai (KHÔNG tự chia lại).

Cần Internet ở máy chạy. Tệp dữ liệu thô được tải riêng;
khi tải ghi checksum và metadata nguồn để truy xuất.
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.request
from pathlib import Path
import pandas as pd
from .common import DATA, EXPECTED, fingerprint, save_json, load_split, versions

BASE = 'https://huggingface.co/datasets/nanyy1025/covid_fake_news/resolve/main/'
SOURCE = 'https://huggingface.co/datasets/nanyy1025/covid_fake_news/tree/main'


def download(data_dir: Path = DATA, force: bool = False):
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'source': SOURCE, 'official_paper':'https://arxiv.org/abs/2011.03327',
                'note':'Bản sao cộng đồng trên Hugging Face; kiểm tra phân bố và checksum trước khi dùng.',
                'versions': versions(), 'files':{}}
    for name, expected in EXPECTED.items():
        target = data_dir / (name+'.csv')
        if force or not target.exists():
            url=BASE+name+'.csv'
            print('Đang tải',url,flush=True)
            err=None
            for attempt in range(3):
                try:
                    req=urllib.request.Request(url, headers={'User-Agent':'student-covid19-nlp-research/1.0'})
                    with urllib.request.urlopen(req,timeout=90) as inp, (data_dir / (name+'.partial')).open('wb') as out:
                        import shutil
                        shutil.copyfileobj(inp,out)
                    (data_dir / (name+'.partial')).replace(target)
                    err=None; break
                except Exception as e:
                    err=e; time.sleep(2 * (attempt+1))
            if err is not None:
                raise RuntimeError(f'Không tải được {name}.csv: {err}; tải thủ công từ {url} vào data/raw/') from err
        df=load_split(name,data_dir)
        if len(df)!=expected:
            raise ValueError(f'{name}: có {len(df)} dòng; cần {expected}. Kiểm tra phiên bản nguồn trước khi dùng.')
        metadata['files'][name]={'rows':len(df),'label_counts':df['label'].value_counts().to_dict(),
                                 'sha256':fingerprint(target),'url':BASE+name+'.csv'}
        print(f'{name:5s}: {len(df):5d} mẫu; {metadata["files"][name]["label_counts"]}')
    save_json(metadata,data_dir.parent / 'dataset_provenance.json')
    print('OK: data/dataset_provenance.json')
    return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true',help='Tải lại các file đã tồn tại')
    args=parser.parse_args()
    download(force=args.force)
