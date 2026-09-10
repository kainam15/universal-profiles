import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from acprof.workloads.cv import CVWorkloadGenerator


class CVWorkloadTests(unittest.TestCase):
    def test_every_task_rejects_unimplemented_batching(self):
        for task in ("image-classification", "mask-generation", "video-classification"):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "batch_size=1"):
                CVWorkloadGenerator("example/model", task, 2)

    def test_zero_shot_labels_are_present_and_recorded(self):
        generator = CVWorkloadGenerator("example/model", "zero-shot-object-detection", 1)
        payload = generator.generate(0.25)
        self.assertEqual(payload["candidate_labels"], ["cat", "dog", "car", "person"])
        self.assertEqual(generator.plan_metadata()["candidate_labels"], payload["candidate_labels"])

    def test_video_uses_distinct_reproducible_frames_at_legacy_resolution(self):
        generator = CVWorkloadGenerator("example/model", "video-classification", 1)
        payload = generator.generate(0.125)
        self.assertEqual(payload, generator.generate(0.125))
        self.assertEqual(len(payload["frames_base64"]), 16)
        self.assertGreater(len(set(payload["frames_base64"])), 1)
        with Image.open(io.BytesIO(base64.b64decode(payload["frames_base64"][0]))) as frame:
            self.assertEqual(frame.size, (28, 28))
        metadata = generator.input_metadata(0.125, payload)
        self.assertEqual(metadata["num_frames"], 16)
        self.assertEqual(len(metadata["frame_sha256"]), 16)
        self.assertIsNone(generator.default_input_scales())

    def test_spec_tracks_local_assets_and_scales_normalized_pose_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            Image.new("RGB", (12, 8), "red").save(path / "source.png")
            (path / "spec.json").write_text(json.dumps({
                "schema_version": 1, "image_path": "source.png",
                "input_scales": [0.5, 1], "boxes": [[0.25, 0.125, 0.5, 0.75]],
                "params": {"dataset_index": 0},
            }))
            generator = CVWorkloadGenerator("example/model", "keypoint-detection", 1,
                                            workload_spec_path=str(path / "spec.json"))
            payload = generator.generate(0.5)
            self.assertEqual(payload["boxes"], [[28, 14, 56, 84]])
            self.assertEqual(generator.default_input_scales(), [0.5, 1])
            self.assertEqual(generator.plan_metadata()["params"], {"dataset_index": 0})
            self.assertEqual(len(generator.plan_metadata()["source_sha256"][0]), 64)

    def test_video_manifest_never_resamples_or_drops_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for index, color in enumerate(("red", "blue")):
                Image.new("RGB", (8, 8), color).save(path / f"{index}.png")
            (path / "spec.json").write_text(json.dumps({"video_frames": ["0.png", "1.png"]}))
            generator = CVWorkloadGenerator("example/model", "video-classification", 1,
                                            workload_spec_path=str(path / "spec.json"))
            self.assertEqual(len(generator.generate(0.1)["frames_base64"]), 2)
            (path / "spec.json").write_text(json.dumps({
                "video_frames": ["0.png", "1.png"], "num_frames": 3,
            }))
            with self.assertRaisesRegex(ValueError, "num_frames"):
                CVWorkloadGenerator("example/model", "video-classification", 1,
                                    workload_spec_path=str(path / "spec.json"))

    def test_bad_scales_are_rejected(self):
        generator = CVWorkloadGenerator("example/model", "image-classification", 1)
        for scale in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                generator.generate(scale)

    def test_spec_rejects_unsupported_or_mismatched_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            for spec in ({"typo": 1}, {"candidate_labels": []}, {"params": {"batch_size": 3}},
                         {"video_frames": ["missing.png"]}, {"input_scales": [0.5, 0.5]}):
                path.write_text(json.dumps(spec))
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    CVWorkloadGenerator("example/model", "zero-shot-image-classification", 1,
                                        workload_spec_path=str(path))


if __name__ == "__main__":
    unittest.main()
