from honesty.data.dataset import HonestyDataset, DatasetTypes
import pytest

@pytest.fixture
def sample_dataset():
    return HonestyDataset(
        data=[
            {"text": "This is a truthful statement.", "label": DatasetTypes.TRUTHFUL},
            {"text": "This is a false statement.", "label": DatasetTypes.FALSE},
        ]
    )

def test_dataset_loading(sample_dataset):
    assert len(sample_dataset.data) == 2
    assert sample_dataset.data[0]["label"] == DatasetTypes.TRUTHFUL

def test_dataset_types():
    assert DatasetTypes.TRUTHFUL.value == "truthful"
    assert DatasetTypes.FALSE.value == "false"