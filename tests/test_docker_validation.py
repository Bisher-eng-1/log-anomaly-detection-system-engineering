import unittest

from scripts.validate_docker import parse_size


class DockerSizeParsingTests(unittest.TestCase):
    def test_parses_binary_mebibytes(self):
        self.assertEqual(parse_size("128MiB"), 128 * 1024 * 1024)

    def test_parses_binary_kibibytes(self):
        self.assertEqual(parse_size("512KiB"), 512 * 1024)

    def test_parses_decimal_megabytes(self):
        self.assertEqual(parse_size("1.5MB"), 1_500_000)

    def test_parses_bytes(self):
        self.assertEqual(parse_size("256B"), 256)

    def test_rejects_unknown_format(self):
        self.assertIsNone(parse_size("not-a-size"))


if __name__ == "__main__":
    unittest.main()
