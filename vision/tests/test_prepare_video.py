import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prepare_video', Path(__file__).parents[1] / 'prepare_video.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class VideoPreparationTest(unittest.TestCase):
    def test_verified_asset_is_offline(self):
        data = b'video fixture'
        entry = {'path': 'assets/test.avi', 'url': 'https://example.invalid/video',
                 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(data)):
                path = module.prepare(root, entry)
            with patch.object(module.urllib.request, 'urlopen', side_effect=AssertionError('network used')):
                self.assertEqual(path, module.prepare(root, entry, offline=True))

    def test_corrupt_download_preserves_existing_file(self):
        entry = {'path': 'test.avi', 'url': 'https://example.invalid/video',
                 'bytes': 5, 'sha256': hashlib.sha256(b'valid').hexdigest()}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'test.avi').write_bytes(b'previous')
            with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    module.prepare(root, entry)
            self.assertEqual(b'previous', (root / 'test.avi').read_bytes())
            with self.assertRaisesRegex(RuntimeError, 'missing or checksum'):
                module.prepare(root, entry, offline=True)


if __name__ == '__main__':
    unittest.main()
