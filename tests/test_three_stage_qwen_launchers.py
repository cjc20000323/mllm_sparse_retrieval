"""Execute all four launchers and parse their Qwen image-limit arguments."""
from pathlib import Path
import shutil
import unittest

import test_three_stage_launcher_paths as fixtures
import test_three_stage_time_launcher as timing
import test_three_stage_retrieval_pipeline as core


@unittest.skipUnless(fixtures.BASH, "Bash is required for launcher integration")
class QwenLauncherTests(unittest.TestCase):
    def test_four_scripts_forward_environment_and_cli_pixel_overrides(self):
        for filename in ("three_stage_retrieval_pipeline.sh", "three_stage_retrieval_pipeline_tbpr.sh",
                         "three_stage_retrieval_pipeline_multimodal.sh", "estimate_three_stage_retrieval_time.sh"):
            with self.subTest(script=filename):
                estimate = filename.startswith("estimate")
                case = timing.TimeLauncherTests() if estimate else fixtures.ThreeStageLauncherPathTests()
                try:
                    case.setUp()
                    case.script = case.root / "scripts" / filename
                    shutil.copyfile(fixtures.PROJECT_ROOT / "scripts" / filename, case.script)
                    result, argv = case.launch(
                        "--retrieval-qwen-max-pixels", "512", "--ranker-qwen-max-pixels=1024",
                        RETRIEVAL_QWEN_MIN_PIXELS="64", RETRIEVAL_QWEN_MAX_PIXELS="256",
                        RANKER_QWEN_MIN_PIXELS="64", RANKER_QWEN_MAX_PIXELS="144",
                        NUM_PROCESSES="1",
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    target = "estimate_three_stage_retrieval_time.py" if estimate else "three_stage_retrieval_pipeline.py"
                    start = next(index for index, value in enumerate(argv) if value.endswith(target))
                    parser = timing.estimate.parser() if estimate else core.pipeline._parser()
                    parsed = parser.parse_args(argv[start + 1:])
                    self.assertEqual((parsed.retrieval_qwen_min_pixels, parsed.retrieval_qwen_max_pixels), (64, 512))
                    self.assertEqual((parsed.ranker_qwen_min_pixels, parsed.ranker_qwen_max_pixels), (64, 1024))
                finally:
                    case.doCleanups()


if __name__ == "__main__":
    unittest.main()
