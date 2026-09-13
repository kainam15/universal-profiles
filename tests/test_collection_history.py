import json
import tempfile
import unittest
from pathlib import Path

from acprof.host.collection_history import (
    COLLECTION_HISTORY_FIELDS,
    COLLECTION_HISTORY_SCHEMA_VERSION,
    append_collection_record,
    empty_collection_history,
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
