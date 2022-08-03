import os
import unittest

from experiment import process_experiment


class TestExperiment(unittest.TestCase):
    files_to_remove = []

    def __init__(self, methodName):
        super().__init__(methodName)

        self.experiment_params = dict(
            output_path="./output",
            data_config="./test-data-config.json",
            experiment_name="ets1_runx1",
            num_layers=2,
            mers=2,
            batch_size=32,
            kernel_widths=[4, 4],
            include_affinities=False,
            pool=False,
            patience=20,
            max_epochs=60,
            debug=True,
        )

    @classmethod
    def tearDownClass(cls):
        for f in cls.files_to_remove:
            try:
                os.remove(f)
            except FileNotFoundError:
                continue

    def test_happy_path(self):
        """Test a happy path."""
        self.files_to_remove += ["./output/task_1.json", "./output/task_1.pdf"]

        process_experiment(job_id=1, **self.experiment_params)

    def test_pool(self):
        """Test that max pooling works."""
        self.files_to_remove += ["./output/task_2.json", "./output/task_2.pdf"]
        params = dict(job_id=2, pool=True)
        params.update(self.experiment_params)

        process_experiment(**params)

    def test_include_affinities(self):
        """Test that including affinities work."""
        self.files_to_remove += ["./output/task_3.json", "./output/task_3.pdf"]
        params = dict(job_id=3, include_affinities=True)
        params.update(self.experiment_params)

        process_experiment(**params)
