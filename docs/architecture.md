# Architecture Overview

This document provides a detailed overview of the architecture of the Model Honesty Research Platform.

## System Components

The platform is composed of two main components, each with a specific role:

### 1. Python Backend

The backend is responsible for machine learning model interactions, evaluations, and training. It provides:

- Dataset curation and management
- Truthfulness evaluation
- Hallucination detection
- Honesty training via reinforcement learning
- Metrics calculation and benchmarking
- RESTful API for frontend communication

#### Key Modules:

- `honesty.data`: Data curation and management
- `honesty.evaluation`: Truthfulness and hallucination evaluation
- `honesty.training`: Reward modeling and reinforcement learning from human feedback (RLHF)
- `honesty.metrics`: Metrics computation and benchmarking
- `honesty.api`: REST API endpoints
- `honesty.utils`: Utilities and helpers

### 2. Rust/Tauri Dashboard

The dashboard provides an interactive user interface for:

- Model and dataset selection
- Evaluation configuration
- Result visualization
- Benchmark comparison
- Training monitoring

#### Key Components:

- `main.rs`: Tauri application entry point
- `api.rs`: Communication with the backend
- `ui/`: React-based user interface with visualization components

## Data Flow

1. **User Interaction**: Users interact with the dashboard to select models and datasets, configure evaluations, and view results.
2. **Evaluation Flow**: The dashboard sends requests to the backend to evaluate selected models on specified datasets.
3. **Training Flow**: Users can initiate training processes for models based on evaluation results.
4. **Benchmarking Flow**: The platform allows users to compare multiple models based on honesty metrics.

## Key Abstractions

### Data Abstractions

- **HonestyDataset**: Common interface for truthfulness and hallucination datasets.
- **TruthfulnessExample**: Structure for truthful/non-truthful statements.
- **HallucinationExample**: Structure for responses with hallucination annotations.

### Evaluation Abstractions

- **TruthfulnessEvaluator**: Evaluates model truthfulness.
- **HallucinationDetector**: Detects hallucinations in model outputs.
- **HonestyMetrics**: Common metrics for honesty evaluation.

### Training Abstractions

- **RewardModelTrainer**: Trains reward models for honesty.
- **RLHFTrainer**: Implements reinforcement learning from human feedback for honesty optimization.

## Communication Protocols

- **Dashboard ↔ Backend**: REST API over HTTP.
- **Backend ↔ Models**: PyTorch/Hugging Face model interface.
- **Dashboard ↔ User**: Interactive UI elements.

## Security Considerations

- **Authentication**: API endpoints require authentication tokens.
- **Model Access**: Access control for proprietary models.
- **Data Privacy**: Secure handling of evaluation data.

## Deployment Architecture

The deployment architecture consists of separate containers for the backend and dashboard services, allowing for scalability and isolation.

## Performance Considerations

- **Parallel Evaluation**: Multiple examples evaluated concurrently.
- **Efficient Model Loading**: Models loaded once and reused.
- **Caching**: Results cached to avoid redundant computation.
- **Optimized Visualizations**: Efficient rendering of large datasets.