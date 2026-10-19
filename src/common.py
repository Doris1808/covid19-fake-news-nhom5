"""Hàm dùng chung: đường dẫn, kiểm tra dữ liệu, metric và lưu JSON.

Quy ước dữ liệu: cột id, tweet, label; label ∈ {real, fake}.
Tuyệt đối không fit vectorizer/scaler bằng validation/test.
"""
from __future__ import annotations
import hashlib
import json
import platform
import random
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
OUT = ROOT / 'outputs'
LABELS = ['fake', 'real']  # Dùng thống nhất cho confusion matrix
EXPECTED = {'train': 6420, 'val': 2140, 'test': 2140}


def mkdir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path):
    mkdir(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load_split(split: str, data_dir: Path = DATA) -> pd.DataFrame:
    if split not in EXPECTED:
        raise ValueError('split phải là train, val hoặc test')
    path = data_dir / f'{split}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Thiếu {path}. Hãy chạy: python -m src.download_data')
    df = pd.read_csv(path, dtype={'id': 'string'})
    if 'text' in df.columns and 'tweet' not in df.columns:
        df = df.rename(columns={'text':'tweet'})
    required = {'tweet','label'}
    if not required.issubset(df.columns):
        raise ValueError(f'{path}: cần các cột tweet,label; nhận {list(df.columns)}')
    if 'id' not in df:
        df.insert(0, 'id', np.arange(1,len(df)+1).astype(str))
    if df['tweet'].isna().any() or df['label'].isna().any():
        raise ValueError(f'{path}: có tweet/label bị thiếu; cần xử lý có biên bản trước khi train')
    df['tweet'] = df['tweet'].astype(str)
    if df.empty or df['tweet'].str.strip().eq('').any():
        raise ValueError(f'{path}: dữ liệu rỗng hoặc có văn bản chỉ chứa khoảng trắng')
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    unknown = set(df['label']) - set(LABELS)
    if unknown:
        raise ValueError(f'{path}: nhãn không hợp lệ: {unknown}')
    return df[['id','tweet','label']].reset_index(drop=True)


def scores(y_true, y_pred) -> dict:
    return {
        'accuracy':float(accuracy_score(y_true,y_pred)),
        'f1_weighted':float(f1_score(y_true,y_pred,average='weighted',zero_division=0)),
        'f1_macro':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
        'f1_fake':float(f1_score(y_true,y_pred,labels=['fake'],average='macro',zero_division=0)),
        'confusion_matrix_fake_real':confusion_matrix(y_true,y_pred,labels=LABELS).tolist(),
        'classification_report':classification_report(y_true,y_pred,labels=LABELS,output_dict=True,zero_division=0),
    }


def fingerprint(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    import sklearn
    import pandas
    return {'python':sys.version.split()[0], 'os':platform.platform(), 'sklearn':sklearn.__version__, 'pandas':pandas.__version__}


def normalize_for_overlap(s):
    return re.sub(r'\s+',' ',str(s).casefold()).strip()
