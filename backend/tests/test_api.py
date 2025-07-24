import pytest
from fastapi.testclient import TestClient
from honesty.api.server import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Model Honesty Research Platform"}

def test_list_models():
    response = client.get("/models")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_register_model():
    model_data = {"name": "Test Model", "description": "A model for testing"}
    response = client.post("/models", json=model_data)
    assert response.status_code == 201
    assert response.json()["name"] == model_data["name"]

def test_get_model():
    model_id = "1"  # Assuming a model with this ID exists
    response = client.get(f"/models/{model_id}")
    assert response.status_code == 200
    assert "name" in response.json()

def test_evaluate_model():
    evaluation_data = {"model_id": "1", "dataset_id": "test_dataset"}
    response = client.post("/evaluations", json=evaluation_data)
    assert response.status_code == 201
    assert "metrics" in response.json()

def test_list_evaluations():
    response = client.get("/evaluations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)