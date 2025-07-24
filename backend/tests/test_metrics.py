import pytest
from honesty.metrics.honesty_metrics import HonestyMetrics

class TestHonestyMetrics:

    def test_calibration_metric(self):
        # Example data for testing
        predictions = [0.1, 0.4, 0.35, 0.8]
        true_labels = [0, 0, 1, 1]
        
        metrics = HonestyMetrics()
        calibration_score = metrics.calculate_calibration(predictions, true_labels)
        
        assert 0 <= calibration_score <= 1

    def test_honesty_score(self):
        # Example data for testing
        model_outputs = ["This is true.", "This is false.", "This is true."]
        ground_truth = [True, False, True]
        
        metrics = HonestyMetrics()
        honesty_score = metrics.calculate_honesty_score(model_outputs, ground_truth)
        
        assert honesty_score >= 0

    def test_benchmark_results(self):
        # Example benchmark data
        benchmark_data = {
            "model_a": {"accuracy": 0.9, "f1_score": 0.85},
            "model_b": {"accuracy": 0.8, "f1_score": 0.75},
        }
        
        metrics = HonestyMetrics()
        results = metrics.benchmark(benchmark_data)
        
        assert len(results) == 2
        assert results["model_a"]["accuracy"] > results["model_b"]["accuracy"]