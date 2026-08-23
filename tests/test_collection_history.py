import json
import tempfile
import unittest
from pathlib import Path

from acprof.host.collection_history import (
    COLLECTION_HISTORY_FIELDS,
    COLLECTION_HISTORY_SCHEMA_VERSION,
    append_collection_record,
    empty_collection_history,
    migrate_legacy_static_meta_history,
    normalize_collection_history,
    write_collection_history_json,
)


class CollectionHistoryTests(unittest.TestCase):
    def test_empty_document_has_stable_schema(self) -> None:
        payload = empty_collection_history()

        self.assertEqual(
            payload["schema_version"],
            COLLECTION_HISTORY_SCHEMA_VERSION,
        )
        for field in COLLECTION_HISTORY_FIELDS:
            self.assertEqual(payload[field], [])

    def test_migrates_legacy_histories_and_drops_last_run_duplicates(self) -> None:
        posthoc_record = {"completed_at": "2026-08-14T14:35:32+08:00"}
        timeout_record = {"completed_at": "2026-08-21T22:29:33+08:00"}
        static_meta = {
            "schema_version": 2,
            "model_name": "example/model",
            "posthoc_profile_history": [posthoc_record],
            "posthoc_profile_last_run": posthoc_record,
            "timeout_retry_history": [timeout_record],
            "timeout_retry_last_run": timeout_record,
        }
        existing = {
            "schema_version": 1,
            "posthoc_profile_history": [posthoc_record],
            "producer": "unit-test",
        }

        cleaned, history = migrate_legacy_static_meta_history(static_meta, existing)

        self.assertEqual(cleaned, {"schema_version": 2, "model_name": "example/model"})
        self.assertEqual(history["posthoc_profile_history"], [posthoc_record])
        self.assertEqual(history["timeout_retry_history"], [timeout_record])
        self.assertEqual(history["quality_retry_history"], [])
        self.assertEqual(history["producer"], "unit-test")

    def test_last_run_is_recovered_when_legacy_history_is_missing(self) -> None:
        record = {"completed_at": "2026-08-22T11:49:30+08:00"}

        _cleaned, history = migrate_legacy_static_meta_history(
            {"quality_retry_last_run": record}
        )

        self.assertEqual(history["quality_retry_history"], [record])

    def test_append_validates_field_and_keeps_native_json_types(self) -> None:
        record = {"retry_rows": 21, "restored": True, "note": None}

        updated = append_collection_record(
            empty_collection_history(),
            "quality_retry_history",
            record,
        )

        self.assertEqual(updated["quality_retry_history"], [record])
        with self.assertRaises(ValueError):
            append_collection_record(updated, "unknown_history", record)

    def test_atomic_writer_emits_valid_json_without_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collection_history.json"
            write_collection_history_json(empty_collection_history(), path)

            payload = json.loads(path.read_text(encoding="utf-8"))
            leftovers = list(path.parent.glob(".collection_history.json.*.tmp"))

        self.assertEqual(payload, normalize_collection_history(payload))
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
