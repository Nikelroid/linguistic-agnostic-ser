import unittest
import torch
import numpy as np
import pandas as pd
from src.models.feature_extractors import TransformerExtractor

class TestPipeline(unittest.TestCase):
    def test_extractor(self):
        # We use a very small model for testing to avoid heavy downloads
        extractor = TransformerExtractor(model_name="hf-internal-testing/tiny-random-wav2vec2", device="cpu")
        dummy_audio = torch.randn(2, 16000 * 2) # 2 samples, 2 seconds
        layers = extractor.extract_from_waveform(dummy_audio)
        self.assertEqual(len(layers), extractor.get_num_layers())
        self.assertEqual(layers[0].shape[0], 2) # Batch size 2

if __name__ == '__main__':
    unittest.main()
