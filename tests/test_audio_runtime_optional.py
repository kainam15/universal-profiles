"""Offline integration checks for the actual audio model APIs in the audio image.

Small random checkpoints exercise loading, preprocessing and waveform inference;
these tests do not measure the quality of pretrained checkpoints.
"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from acprof.container.handlers.audio import AudioHandler


_HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "scipy"))


@unittest.skipUnless(_HAS_RUNTIME, "audio image dependencies are unavailable")
class AudioRuntimeTests(unittest.TestCase):
    def test_wav2vec_asr_and_classification_run_offline(self):
        from transformers import (
            Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
            Wav2Vec2ForCTC, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor,
        )

        handler = AudioHandler()
        for task, model_class in (
            ("automatic-speech-recognition", Wav2Vec2ForCTC),
            ("audio-classification", Wav2Vec2ForSequenceClassification),
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                config = Wav2Vec2Config(
                    vocab_size=5, hidden_size=8, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=16,
                    conv_dim=(4, 4, 4), conv_stride=(4, 4, 4), conv_kernel=(8, 4, 4),
                    num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
                    feat_extract_norm="layer", mask_time_prob=0.0,
                    classifier_proj_size=8, num_labels=2, pad_token_id=0,
                )
                model_class(config).save_pretrained(directory)
                feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)
                if task == "automatic-speech-recognition":
                    vocab_file = Path(directory) / "vocab.json"
                    vocab_file.write_text(json.dumps({"<pad>": 0, "<unk>": 1, "|": 2, "a": 3, "b": 4}))
                    tokenizer = Wav2Vec2CTCTokenizer(vocab_file=str(vocab_file))
                    Wav2Vec2Processor(feature_extractor, tokenizer).save_pretrained(directory)
                else:
                    feature_extractor.save_pretrained(directory)
                context = handler.load(directory, task, "transformers_pipeline", "cpu")
                processed = handler.preprocess(context, {
                    "audio_samples": np.sin(np.arange(1600) * 0.1).tolist(), "sample_rate": 16000,
                })
                output = handler.postprocess(context, handler.predict(context, processed))
                if task == "automatic-speech-recognition":
                    self.assertEqual(output["output_type"], "transcription")
                    self.assertIsInstance(output["text"], str)
                else:
                    self.assertEqual(output["output_type"], "classification")
                    self.assertGreater(output["n_results"], 0)

    def test_encodec_and_dac_load_and_reconstruct_offline(self):
        import transformers

        configurations = [
            ("Encodec", transformers.EncodecConfig(
                hidden_size=8, num_filters=4, num_residual_layers=1,
                upsampling_ratios=[2, 2], num_lstm_layers=1,
                codebook_size=16, target_bandwidths=[24.0], sampling_rate=24000,
            )),
            ("Dac", transformers.DacConfig(
                encoder_hidden_size=4, decoder_hidden_size=16,
                downsampling_ratios=[2, 2], n_codebooks=2, codebook_size=16,
                codebook_dim=2, sampling_rate=16000,
            )),
        ]
        handler = AudioHandler()
        for family, config in configurations:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                model = getattr(transformers, family + "Model")(config)
                model.save_pretrained(directory)
                processor = getattr(transformers, family + "FeatureExtractor")(
                    sampling_rate=config.sampling_rate
                )
                processor.save_pretrained(directory)
                context = handler.load(directory, "audio-to-audio", "transformers_model", "cpu")
                processed = handler.preprocess(context, {
                    "audio_samples": np.sin(np.arange(1600) * 0.1).tolist(), "sample_rate": 16000,
                })
                output = handler.postprocess(context, handler.predict(context, processed))
                self.assertEqual(output["output_type"], "audio")
                self.assertEqual(output["audio_sample_rate"], config.sampling_rate)
                self.assertAlmostEqual(output["audio_duration_s"], 0.1, places=3)

    def test_vits_text_to_speech_and_text_to_audio_generate_waveform_offline(self):
        from transformers import VitsConfig, VitsModel, VitsTokenizer

        handler = AudioHandler()
        with tempfile.TemporaryDirectory() as directory:
            vocab = {character: index for index, character in enumerate("_? abcdefghijklmnopqrstuvwxyz")}
            vocab_path = Path(directory) / "vocab.json"
            vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
            tokenizer = VitsTokenizer(vocab_file=str(vocab_path), phonemize=False)
            tokenizer.save_pretrained(directory)
            config = VitsConfig(
                vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, ffn_dim=16, flow_size=8, spectrogram_bins=8,
                use_stochastic_duration_prediction=False,
                upsample_initial_channel=16, upsample_rates=[2, 2],
                upsample_kernel_sizes=[4, 4], resblock_kernel_sizes=[3],
                resblock_dilation_sizes=[[1, 3, 5]],
                duration_predictor_filter_channels=8, prior_encoder_num_flows=1,
                prior_encoder_num_wavenet_layers=1, posterior_encoder_num_wavenet_layers=1,
            )
            VitsModel(config).save_pretrained(directory)
            for task in ("text-to-speech", "text-to-audio"):
                with self.subTest(task=task):
                    context = handler.load(directory, task, "transformers_pipeline", "cpu")
                    processed = handler.preprocess(context, {"text": "hello world"})
                    output = handler.postprocess(context, handler.predict(context, processed))
                    self.assertEqual(output["output_type"], "audio")
                    self.assertGreater(output["audio_num_samples"], 0)
                    self.assertEqual(output["audio_sample_rate"], config.sampling_rate)
                    self.assertGreater(processed["_effective_input_scale"], 0)


if __name__ == "__main__":
    unittest.main()
