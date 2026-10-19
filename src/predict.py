"""Nạp checkpoint cục bộ để dự đoán; không tải mô hình gốc thay thế."""
from pathlib import Path


def load_model(model_dir):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = Path(model_dir)
    if not (path/'config.json').exists():
        raise FileNotFoundError(f'Chưa có checkpoint tại {path}. Cần train rồi tải thư mục model về.')
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    if set(model.config.id2label.values()) != {'fake', 'real'}:
        raise ValueError('Checkpoint không có ánh xạ fake/real của đồ án.')
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model, max_length=128):
    import torch
    if not text.strip():
        raise ValueError('Vui lòng nhập văn bản.')
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    with torch.inference_mode():
        probs = model(**tokens).logits.softmax(dim=-1)[0].tolist()
    return {model.config.id2label[i]: p for i, p in enumerate(probs)}
