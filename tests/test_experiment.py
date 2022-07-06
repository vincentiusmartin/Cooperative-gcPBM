import os
import unittest

from experiment import process_experiment_architecture_model


class TestExperiment(unittest.TestCase):
    files_to_remove = []

    @classmethod
    def tearDownClass(cls):
        for f in cls.files_to_remove:
            os.remove(f)

    def test_happy_path(self):
        self.files_to_remove += ["./output/task_1.json", "./output/task_1.pdf"]

        process_experiment_architecture_model(
            job_id=1,
            output_path="./output",
            data_config="./test-data-config.json",
            experiment_name="ets1_runx1",
            num_layers=2,
            mers=1,
            batch_size=32,
            kernel_widths=[8, 8],
            include_affinities=True,
            patience=20,
            max_epochs=50,
            debug=True)
