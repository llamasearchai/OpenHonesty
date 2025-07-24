# Usage Documentation for Model Honesty Research Platform

## Overview

The Model Honesty Research Platform is designed to evaluate and improve the truthfulness and honesty of language models. This document provides guidance on how to use the platform effectively.

## Getting Started

### Prerequisites

Before using the platform, ensure that you have the following installed:

- Python 3.10 or higher
- Rust 1.68 or higher
- Node.js 18 or higher
- Docker (optional, for containerized setup)

### Installation

#### Using Docker (Recommended)

To quickly set up the platform, you can use Docker. Follow these steps:

1. Clone the repository:

   git clone <repository-url>

2. Navigate to the project directory:

   cd honesty-platform

3. Build and start the services:

   docker-compose up -d

4. Access the dashboard at:

   http://localhost:7113

#### Manual Installation

If you prefer to install the platform manually, follow these steps:

1. **Backend Installation**

   Navigate to the backend directory and install dependencies using Poetry:

   cd backend
   poetry install

2. **Dashboard Installation**

   Navigate to the dashboard directory and build the Rust application:

   cd dashboard
   cargo build --release

   Then, navigate to the UI directory and install dependencies:

   cd ui
   npm install
   npm run build

### Running the Platform

To run the platform, follow these steps:

1. **Start the Backend**

   In the backend directory, run the FastAPI server:

   poetry run uvicorn honesty.api.server:app --host 0.0.0.0 --port 8000

2. **Start the Dashboard**

   In the dashboard directory, run the Tauri application:

   cargo run --release

### Using the Dashboard

Once the platform is running, you can interact with the dashboard:

1. **Select a Model**: Choose a language model from the available options.
2. **Select a Dataset**: Choose a dataset for evaluation.
3. **Run Evaluations**: Initiate evaluations to assess model truthfulness or detect hallucinations.
4. **View Results**: Explore the evaluation metrics and visualizations provided by the dashboard.
5. **Compare Models**: Use the benchmarking feature to compare multiple models based on honesty metrics.

### API Usage

The platform also provides a RESTful API for programmatic access. You can interact with the API using HTTP requests. Here’s an example of how to evaluate a model:

```python
import requests

response = requests.post(
    "http://localhost:8000/evaluations",
    json={"model_id": "your_model_id", "dataset_id": "your_dataset_id"}
)

evaluation = response.json()
print(f"Accuracy: {evaluation['metrics']['accuracy']}")
```

## Conclusion

This documentation provides a comprehensive guide to using the Model Honesty Research Platform. For further details on specific features, refer to the API documentation and architecture overview.