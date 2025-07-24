import pytest
from honesty.evaluation.truthfulness import TruthfulnessEvaluator
from honesty.evaluation.hallucination import HallucinationDetector

class TestTruthfulnessEvaluator:
    def setup_method(self):
        self.evaluator = TruthfulnessEvaluator()

    def test_evaluate_truthfulness(self):
        example = "The sky is blue."
        result = self.evaluator.evaluate(example)
        assert result is not None
        assert isinstance(result, dict)
        assert "truthfulness_score" in result

class TestHallucinationDetector:
    def setup_method(self):
        self.detector = HallucinationDetector()

    def test_detect_hallucination(self):
        example = "The moon is made of cheese."
        result = self.detector.detect(example)
        assert result is not None
        assert isinstance(result, dict)
        assert "hallucination_detected" in result
        assert isinstance(result["hallucination_detected"], bool)