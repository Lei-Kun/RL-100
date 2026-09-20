from typing import Dict

from rl_100.env_runner.base_runner import BaseRunner


class OfflineRunner(BaseRunner):
    """Minimal runner for offline training without a simulator or robot."""

    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)
        self.env = None

    def run(self, policy, **kwargs) -> Dict:
        return {
            'test_mean_score': 0.0,
            'mean_returns': 0.0,
        }
