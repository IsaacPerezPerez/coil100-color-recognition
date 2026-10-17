import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.build_review_zip import build_archive


class ReviewPackageTests(unittest.TestCase):
    def test_archive_preserves_content_and_verifies_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "README.md"
            source.write_text("Proyecto de revisión.", encoding="utf-8")
            result = build_archive(root, root / "review.zip", [source])
            self.assertTrue(result["crc_verified"])
            self.assertTrue(result["content_hashes_verified"])
            with ZipFile(root / "review.zip") as archive:
                self.assertEqual(archive.read("README.md"), source.read_bytes())
                self.assertIn("MANIFEST.json", archive.namelist())
            self.assertFalse((root / "review.zip.partial").exists())
            with self.assertRaises(ValueError):
                build_archive(root, root / "review.zip", [source])

    def test_raw_dataset_and_personal_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw/image.png"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"excluded")
            with self.assertRaises(ValueError):
                build_archive(root, root / "review.zip", [raw])
            note = root / "note.md"
            note.write_text("C:" + "/Users/" + "example/Desktop/file", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_archive(root, root / "review.zip", [note])
