"""Huấn luyện DistilBERT một seed; chỉ dùng Validation để chọn checkpoint."""
import argparse
import json
import time
import subprocess
import sys
from pathlib import Path
import numpy as np
from .common import ROOT, load_split, scores, save_json, fingerprint, versions


def run(config_path, seed, data_dir, output, smoke=False, resume=None):
    import torch
    import transformers
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    from torch.utils.data import Dataset
    cfg = json.loads(Path(config_path).read_text())
    if seed not in cfg['seeds']:
        raise ValueError('Seed phải thuộc danh sách trong config.')
    output, data_dir = Path(output), Path(data_dir)
    if (output / 'completed.flag').exists():
        raise FileExistsError('Lượt chạy đã hoàn thành. Chọn thư mục mới để giữ kết quả cũ.')
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Thư mục đã có dữ liệu; dùng --resume hoặc chọn thư mục mới.')
    output.mkdir(parents=True, exist_ok=True)
    train, val = load_split('train', data_dir), load_split('val', data_dir)
    if smoke:
        train = train.groupby('label', group_keys=False).head(16).reset_index(drop=True)
        val = val.groupby('label', group_keys=False).head(8).reset_index(drop=True)
    hashes = {s: fingerprint(data_dir / f'{s}.csv') for s in ['train', 'val']}
    protocol = dict(cfg, seed=seed, smoke=smoke, dataset_sha256=hashes)
    if resume:
        previous = json.loads((output / 'run_config.json').read_text())
        if protocol != previous:
            raise ValueError('Config, seed hoặc dữ liệu khác lượt chạy cũ; không thể resume.')
    save_json(protocol, output / 'run_config.json')
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    class TextDataset(Dataset):
        def __init__(self, frame):
            self.tokens = tokenizer(frame.tweet.tolist(), truncation=True,
                                    max_length=cfg['max_length'])
            self.labels = [0 if x == 'fake' else 1 for x in frame.label]
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, index):
            return {**{k: v[index] for k, v in self.tokens.items()},
                    'labels': self.labels[index]}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=2,
        id2label={0: 'fake', 1: 'real'}, label2id={'fake': 0, 'real': 1})
    def compute_metrics(pred):
        y = ['fake' if x == 0 else 'real' for x in pred.label_ids]
        p = ['fake' if x == 0 else 'real' for x in np.argmax(pred.predictions, axis=-1)]
        return {k: v for k, v in scores(y, p).items() if isinstance(v, float)}

    args = TrainingArguments(
        output_dir=str(output / 'checkpoints'),
        num_train_epochs=1 if smoke else cfg['epochs'],
        learning_rate=cfg['learning_rate'],
        per_device_train_batch_size=cfg['train_batch_size'],
        per_device_eval_batch_size=cfg['eval_batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        weight_decay=cfg['weight_decay'], warmup_ratio=cfg['warmup_ratio'],
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='f1_weighted', greater_is_better=True,
        save_total_limit=2, seed=seed, data_seed=seed,
        fp16=False, report_to='none', optim='adamw_torch',
        logging_steps=10, dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=TextDataset(train),
                      eval_dataset=TextDataset(val), processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    device = trainer.args.device
    gpu_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)
    environment = dict(versions(), torch=torch.__version__, transformers=transformers.__version__,
                       device=str(device), gpu_name=gpu_name, n_gpu=trainer.args.n_gpu)
    save_json(environment, output / 'environment.json')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    (output / 'pip_freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=resume)
    except Exception as error:
        save_json({'error': repr(error), 'elapsed_seconds_this_session': time.perf_counter()-start},
                  output / 'failure.json')
        raise
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None
    # Trainer đã nạp checkpoint tốt nhất trên Validation trước khi export.
    trainer.save_model(str(output / 'model'))
    tokenizer.save_pretrained(output / 'model')
    prediction = trainer.predict(TextDataset(val)).predictions.argmax(axis=-1)
    decoded = ['fake' if x == 0 else 'real' for x in prediction]
    frame = val.assign(predicted_label=decoded)
    frame.to_csv(output / 'val_predictions.csv', index=False)
    frame[frame.label != frame.predicted_label].to_csv(output / 'val_errors.csv', index=False)
    result = {
        'architecture': 'distilbert', 'seed': seed, 'smoke': smoke,
        'val': scores(val.label.tolist(), decoded), 'n_train': len(train), 'n_val': len(val),
        'train_seconds_this_session': seconds, 'resumed': bool(resume),
        'peak_cuda_allocated_gib': peak, 'environment': environment,
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'effective_batch_size': cfg['train_batch_size'] * cfg['gradient_accumulation_steps'] * max(1, trainer.args.n_gpu),
        'dataset_sha256': hashes,
    }
    save_json(trainer.state.log_history, output / 'log_history.json')
    save_json(result, output / 'metrics.json')
    (output / 'completed.flag').write_text('completed\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'configs/distilbert.json'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data-dir', default=str(ROOT/'data/raw'))
    p.add_argument('--out-dir')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', help='Đường dẫn checkpoint-N của cùng lượt chạy')
    a = p.parse_args()
    output = a.out_dir or str(ROOT/'outputs'/('smoke' if a.smoke else 'distilbert')/f'seed_{a.seed}')
    run(a.config, a.seed, a.data_dir, output, a.smoke, a.resume)
