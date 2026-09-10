"""Offline CPU checks against real Transformers using random tiny model weights."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from acprof.container.handlers.cv import CVHandler
from acprof.workloads.cv import CVWorkloadGenerator


_RUNTIME_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


@unittest.skipUnless(_RUNTIME_AVAILABLE, "requires the CV image runtime")
class CVRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        torch.set_num_threads(1)

    def _exercise(self, model, processor, task, spec=None):
        handler = CVHandler()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            model.save_pretrained(snapshot)
            processor.save_pretrained(snapshot)
            context = handler.load(str(snapshot), task, "transformers_model", "cpu")
            manifest = None
            if spec is not None:
                manifest = Path(directory) / "workload.json"
                manifest.write_text(json.dumps(spec))
            generator = CVWorkloadGenerator("local/tiny", task, 1,
                                            workload_spec_path=str(manifest) if manifest else None)
            processed = handler.preprocess(context, generator.generate(0.25))
            result = handler.predict(context, processed)
            return handler.postprocess(context, result)

    def test_videomae_loads_local_snapshot_and_consumes_every_frame(self):
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

        model = VideoMAEForVideoClassification(VideoMAEConfig(
            image_size=32, patch_size=16, num_frames=4, tubelet_size=2,
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
            num_labels=3, id2label={0: "zero", 1: "one", 2: "two"},
        ))
        processor = VideoMAEImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
        result = self._exercise(model, processor, "video-classification", {"num_frames": 4, "params": {"top_k": 2}})
        self.assertEqual(result["output_type"], "classification")
        self.assertEqual(result["n_results"], 2)
        self.assertEqual(len(result["classifications"]), 2)
        self.assertTrue(all(0 <= record["score"] <= 1 for record in result["classifications"]))

    def test_superpoint_loads_local_snapshot_and_counts_valid_keypoints(self):
        from transformers import SuperPointConfig, SuperPointForKeypointDetection, SuperPointImageProcessor

        model = SuperPointForKeypointDetection(SuperPointConfig(
            encoder_hidden_sizes=[8, 8, 16, 16], decoder_hidden_size=16,
            descriptor_decoder_dim=16, max_keypoints=20,
        ))
        processor = SuperPointImageProcessor(size={"height": 64, "width": 64})
        result = self._exercise(model, processor, "keypoint-detection")
        self.assertEqual(result["output_type"], "keypoints")
        self.assertEqual(result["n_results"], 1)
        self.assertLessEqual(result["keypoint_count"], 20)

    def test_sam_mask_pipeline_returns_a_mask_count(self):
        from transformers import SamConfig, SamImageProcessor, SamMaskDecoderConfig, SamModel, SamPromptEncoderConfig, SamVisionConfig

        model = SamModel(SamConfig(
            vision_config=SamVisionConfig(hidden_size=32, output_channels=32, num_hidden_layers=1,
                                          num_attention_heads=4, image_size=64, patch_size=16,
                                          global_attn_indexes=[0], num_pos_feats=16, mlp_dim=64),
            prompt_encoder_config=SamPromptEncoderConfig(hidden_size=32, image_size=64, patch_size=16),
            mask_decoder_config=SamMaskDecoderConfig(hidden_size=32, mlp_dim=64, num_hidden_layers=1,
                                                     num_attention_heads=4, iou_head_hidden_dim=32),
        ))
        processor = SamImageProcessor(size={"longest_edge": 64}, pad_size={"height": 64, "width": 64})
        result = self._exercise(model, processor, "mask-generation", {"params": {
            "points_per_batch": 4, "points_per_crop": 2, "pred_iou_thresh": 0.0,
            "stability_score_thresh": 0.0,
        }})
        self.assertEqual(result["output_type"], "masks")
        self.assertIsInstance(result["n_results"], int)

    def test_owlvit_zero_shot_pipeline_uses_candidate_labels(self):
        from transformers import CLIPTokenizer, OwlViTConfig, OwlViTForObjectDetection, OwlViTImageProcessor, OwlViTTextConfig, OwlViTVisionConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vocab.json").write_text(json.dumps({"<pad>": 0, "<|startoftext|>": 1, "a</w>": 2, "<|endoftext|>": 3}))
            (root / "merges.txt").write_text("#version: 0.2\n")
            tokenizer = CLIPTokenizer(vocab_file=str(root / "vocab.json"), merges_file=str(root / "merges.txt"),
                                      model_max_length=16)
            model = OwlViTForObjectDetection(OwlViTConfig(
                text_config=OwlViTTextConfig(vocab_size=4, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
                    bos_token_id=1, eos_token_id=3, pad_token_id=0).to_dict(),
                vision_config=OwlViTVisionConfig(hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16).to_dict(),
                projection_dim=32,
            ))
            snapshot = root / "snapshot"
            model.save_pretrained(snapshot)
            tokenizer.save_pretrained(snapshot)
            OwlViTImageProcessor(size={"height": 32, "width": 32}).save_pretrained(snapshot)
            handler = CVHandler()
            context = handler.load(str(snapshot), "zero-shot-object-detection", "transformers_pipeline", "cpu")
            payload = CVWorkloadGenerator("local/tiny", "zero-shot-object-detection", 1).generate(0.25)
            payload["candidate_labels"] = ["a", "cat"]
            payload["params"] = {"threshold": 0.0}
            raw = handler.predict(context, handler.preprocess(context, payload))
            self.assertTrue(raw)
            self.assertTrue({record["label"] for record in raw}.issubset({"a", "cat"}))
            self.assertEqual(handler.postprocess(context, raw)["output_type"], "detection")

    @unittest.skipUnless(importlib.util.find_spec("scipy") is not None, "VitPose requires scipy")
    def test_vitpose_loads_local_snapshot_and_uses_every_coco_box(self):
        from transformers import VitPoseBackboneConfig, VitPoseConfig, VitPoseForPoseEstimation, VitPoseImageProcessor

        backbone = VitPoseBackboneConfig(
            image_size=[32, 32], patch_size=[16, 16], hidden_size=32,
            num_hidden_layers=1, num_attention_heads=2, part_features=8,
        )
        model = VitPoseForPoseEstimation(VitPoseConfig(backbone_config=backbone, num_labels=17))
        processor = VitPoseImageProcessor(size={"height": 32, "width": 32})
        result = self._exercise(model, processor, "keypoint-detection", {
            "boxes": [[0, 0, 0.5, 1], [0.5, 0, 0.5, 1]], "params": {"dataset_index": 0},
        })
        self.assertEqual(result["output_type"], "keypoints")
        self.assertEqual(result["n_results"], 2)
        self.assertEqual(result["keypoint_count"], 34)


if __name__ == "__main__":
    unittest.main()
