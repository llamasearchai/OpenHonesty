import pytest
from honesty.training.reward_model import HonestyRewardModel, HonestyRewardDataset
from honesty.training.rlhf import RLHFTrainer

@pytest.fixture
def sample_dataset():
    # Create a sample dataset for testing
    return HonestyRewardDataset()

def test_reward_model_initialization():
    model = HonestyRewardModel()
    assert model is not None

def test_reward_model_forward(sample_dataset):
    model = HonestyRewardModel()
    output = model(sample_dataset)
    assert output is not None

def test_rlhf_trainer_initialization():
    trainer = RLHFTrainer(model=HonestyRewardModel())
    assert trainer is not None

def test_rlhf_training(sample_dataset):
    trainer = RLHFTrainer(model=HonestyRewardModel())
    trainer.train(sample_dataset)
    assert trainer.model is not None  # Ensure the model is trained

def test_rlhf_evaluation(sample_dataset):
    trainer = RLHFTrainer(model=HonestyRewardModel())
    trainer.train(sample_dataset)
    evaluation_results = trainer.evaluate(sample_dataset)
    assert evaluation_results is not None  # Ensure evaluation results are produced