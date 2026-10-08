"""Board failure gates run without FPGA access or simulated measurements."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HARDWARE = Path(__file__).resolve().parents[1]
ROOT = HARDWARE.parent
sys.path.insert(0, str(HARDWARE))
import run_board


class VerificationDevice:
    """Only the interface needed to test verification process status."""

    mismatch = False

    def load_tables(self, package):
        pass

    def load_inputs(self, package, batch):
        return 0

    def verify_slot(self, package, slot, index, variant, check_z1):
        errors = ['integer output mismatch'] if self.mismatch else []
        return errors, {'mf': int(package.exp['mf'][index])}, 0.001, {}

    def close(self):
        pass


class BoardGateTest(unittest.TestCase):
    def test_portable_build_help_needs_only_standard_library(self):
        result = subprocess.run(
            [sys.executable, '-S', str(HARDWARE / 'build.py'), '--help'],
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--variant', result.stdout)

    def test_zero_verification_count_is_rejected_before_board_access(self):
        with patch.object(sys, 'argv', ['run_board.py', '--variant', 'mf', '--verify', '0']), \
                patch.object(run_board, 'preflight') as preflight, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_board.main()
        self.assertEqual(error.exception.code, 2)
        preflight.assert_not_called()

    def test_corrupted_package_is_rejected_before_board_access(self):
        output = ROOT / 'outputs' / 'hardware'
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            package = Path(temp)
            (package / 'weights.bin').write_bytes(b'changed')
            (package / 'sha256.json').write_text(json.dumps({'weights.bin': '0' * 64}))
            args = ['run_board.py', '--variant', 'mf', '--power',
                    '--package', package.relative_to(ROOT).as_posix()]
            with patch.object(sys, 'argv', args), patch.object(run_board, 'preflight') as preflight:
                with self.assertRaisesRegex(ValueError, 'Fixture hash mismatch'):
                    run_board.main()
            preflight.assert_not_called()

    def test_verification_mismatch_fails_and_preserves_evidence(self):
        output = ROOT / 'outputs' / 'hardware'
        output.mkdir(parents=True, exist_ok=True)
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory(dir=output) as temp:
                work = Path(temp)
                build = work / 'build' / 'mf'
                build.mkdir(parents=True)
                (build / 'build_receipt_mf.json').write_text('{}')
                destination = work / 'verification'
                device = VerificationDevice()
                device.mismatch = mismatch
                args = ['run_board.py', '--variant', 'mf', '--verify', '1',
                        '--build-root', build.parent.relative_to(ROOT).as_posix(),
                        '--out', destination.relative_to(ROOT).as_posix()]
                with patch.object(sys, 'argv', args), \
                        patch.object(run_board, 'preflight', return_value=({}, 100000000)), \
                        patch.object(run_board, 'Crisp', return_value=device), \
                        patch.object(run_board, 'check_identity'), \
                        contextlib.redirect_stdout(io.StringIO()):
                    if mismatch:
                        with self.assertRaisesRegex(RuntimeError, 'Board verification failed'):
                            run_board.main()
                    else:
                        run_board.main()
                record = json.loads((destination / 'board_verification.json').read_text())
                self.assertEqual(record['bit_exact'], [not mismatch])
                self.assertEqual((destination / 'failure.json').exists(), mismatch)
                self.assertTrue((destination / 'board_mf_results.zip').is_file())


if __name__ == '__main__':
    unittest.main()
