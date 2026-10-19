"""Kiểm tra phần mềm bằng dữ liệu giả, không phải kết quả nghiên cứu."""
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.common import load_split, scores, save_json
from src.summarize import collect

class CoreTests(unittest.TestCase):
    def test_label_order_and_metrics(self):
        m = scores(['fake','real','real'], ['fake','fake','real'])
        self.assertAlmostEqual(m['accuracy'], 2/3)
        self.assertEqual(m['confusion_matrix_fake_real'], [[1,0],[1,1]])
        self.assertAlmostEqual(m['f1_fake'], 2/3)

    def test_bad_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for text, label in [('hello','unknown'), ('   ','fake')]:
                pd.DataFrame([{'tweet':text, 'label':label}]).to_csv(root/'train.csv', index=False)
                with self.assertRaises(ValueError):
                    load_split('train', root)

    def test_exclude_smoke_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_json({'smoke':True}, root/'seed_42/metrics.json')
            save_json({'smoke':False}, root/'seed_123/metrics.json')
            frame = collect(root)
            self.assertEqual(len(frame), 0)

if __name__ == '__main__':
    unittest.main()
