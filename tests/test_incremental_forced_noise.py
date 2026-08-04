import unittest

import numpy as np
import pandas as pd

from incremental_clustering import _apply_global_forced_noise


class GlobalForcedNoiseTest(unittest.TestCase):
    def test_reranks_the_complete_accumulated_dataset(self) -> None:
        sample_count = 200
        assignments = pd.DataFrame(
            {
                "id": [f"doc-{index:03d}" for index in range(sample_count)],
                "noise_score": np.linspace(0.0, 1.0, sample_count),
                "is_noise": False,
                "is_natural_noise": False,
                "is_forced_noise": False,
                "boundary_level": -1,
                "noise_level": -1,
                "level_1_cluster": 0,
            }
        )

        result = _apply_global_forced_noise(
            assignments,
            forced_noise_ratio=0.01,
        )

        self.assertEqual(int(result["is_forced_noise"].sum()), 2)
        self.assertEqual(
            set(result.loc[result["is_forced_noise"], "id"]),
            {"doc-198", "doc-199"},
        )
        self.assertTrue(
            (result.loc[result["is_forced_noise"], "cluster"] == -1).all()
        )


if __name__ == "__main__":
    unittest.main()
