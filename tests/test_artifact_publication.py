import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from artifact_publication import fresh_artifact_directory


class ArtifactPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "new"

    def test_destination_invisible_until_complete(self):
        with fresh_artifact_directory(self.output) as stage:
            (stage / "complete.txt").write_text("complete synthetic artifact")
            self.assertFalse(self.output.exists())
        self.assertEqual((self.output / "complete.txt").read_text(), "complete synthetic artifact")
        self.assertEqual(list(self.root.iterdir()), [self.output])

    def test_concurrent_cooperating_writer_fails_without_disturbing_first(self):
        with fresh_artifact_directory(self.output) as stage:
            with self.assertRaises(FileExistsError):
                with fresh_artifact_directory(self.output):
                    self.fail("second writer should not start")
            (stage / "first").write_text("first writer")
        self.assertTrue((self.output / "first").is_file())

    def test_destination_appearing_during_build_is_not_replaced(self):
        with self.assertRaises(FileExistsError):
            with fresh_artifact_directory(self.output) as stage:
                (stage / "staged").write_text("synthetic")
                self.output.mkdir()
                (self.output / "external").write_text("preserve this")
        self.assertEqual([p.name for p in self.output.iterdir()], ["external"])
        self.assertFalse(list(self.root.glob(".new.*")))

    def test_build_exception_cleans_stage_and_leaves_no_package(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
            with fresh_artifact_directory(self.output) as stage:
                (stage / "partial").write_text("synthetic")
                raise RuntimeError("synthetic interruption")
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
