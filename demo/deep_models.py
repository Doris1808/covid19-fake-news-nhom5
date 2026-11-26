"""Optional CPU inference for native Streamlit (Community Cloud or a local server)."""
import hashlib
from html import unescape
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZipFile

from filelock import FileLock
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
BERT = "BERT-base · seed 123"
BART = "BART-MNLI · zero-shot"
DISTIL = "DistilBERT · checkpoint nhóm"
ROBERTA = "RoBERTa gốc · checkpoint nhóm"
BERT_URL = ("https://github.com/Doris1808/covid19-fake-news-nhom5/releases/download/"
            "v1.0-nop-bai/Nhom5_COVID19_FakeNews_DayDu.zip")
ZIP_SHA256 = "f746e353f0ab3a6684cde5ab107ab25a3831a7f6ede0dd1571f64bda09e305d1"
DISTIL_PATH = Path(os.environ.get("NHOM5_DISTILBERT", ROOT / "outputs/distilbert/seed_42/model"))
ROBERTA_PATH = Path(os.environ.get("NHOM5_ROBERTA", ROOT / "outputs/roberta/seed_123/model"))


def ready(folder):
    return all((folder / name).is_file() for name in ("model.safetensors", "config.json"))


def native_options():
    names = [BERT, BART]
    if ready(DISTIL_PATH):
        names.append(DISTIL)
    if ready(ROBERTA_PATH):
        names.append(ROBERTA)
    return names


def bert_checkpoint():
    bundled = ROOT / "members/ly_bert/seed123"
    if ready(bundled):
        return bundled
    home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nhom5-streamlit"
    home.mkdir(parents=True, exist_ok=True)
    target = home / "bert-seed123-v1"
    with FileLock(str(home / "bert.lock"), timeout=240):
        if ready(target):
            return target
        with TemporaryDirectory(dir=home) as temp:
            archive = Path(temp) / "source.zip"
            with urlopen(BERT_URL, timeout=90) as response, archive.open("wb") as dest:
                shutil.copyfileobj(response, dest)
            digest = hashlib.sha256()
            with archive.open("rb") as source:
                for block in iter(lambda: source.read(2**20), b""):
                    digest.update(block)
            if digest.hexdigest() != ZIP_SHA256:
                raise ValueError("Bản tải BERT chưa khớp bản nộp v1.0 của nhóm.")
            checkpoint = Path(temp) / "model"
            checkpoint.mkdir()
            with ZipFile(archive) as zipped:
                for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
                    suffix = "members/ly_bert/seed123/" + name
                    entries = [p for p in zipped.namelist() if p == suffix or p.endswith("/" + suffix)]
                    if len(entries) != 1:
                        raise ValueError("Thiếu tệp của checkpoint BERT: " + name)
                    with zipped.open(entries[0]) as source, (checkpoint / name).open("wb") as dest:
                        shutil.copyfileobj(source, dest)
            checkpoint.rename(target)
    return target


@st.cache_resource(max_entries=1, show_spinner=False)
def load_classifier(name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    if name == BART:
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    folder = bert_checkpoint() if name == BERT else {DISTIL: DISTIL_PATH, ROBERTA: ROBERTA_PATH}[name]
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True).eval()
    if {v.lower() for v in model.config.id2label.values()} != {"real", "fake"}:
        raise ValueError("Checkpoint thiếu ánh xạ REAL/FAKE của nhóm.")
    return tokenizer, model


def classify(name, text):
    import torch
    classifier = load_classifier(name)
    if name == BART:
        cleaned = re.sub(r"@\w+", " ", re.sub(r"https?://\S+", " ", unescape(text)))
        result = classifier(" ".join(cleaned.split()) or ".", candidate_labels=["real news", "fake news"],
                            hypothesis_template="This is {}.")
        p_fake = dict(zip(result["labels"], result["scores"]))["fake news"]
        return ("fake", p_fake) if p_fake > 0.5 else ("real", 1 - p_fake)
    tokenizer, model = classifier
    inputs = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    with torch.inference_mode():
        probabilities = model(**inputs).logits[0].softmax(-1)
    index = probabilities.argmax().item()
    return model.config.id2label[index].lower(), probabilities[index].item()
