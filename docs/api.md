# API Documentation

This document describes the RESTful API provided by the Model Honesty Research Platform.

## Base URL

All API endpoints are relative to the base URL:

http://localhost:8000

## Authentication

API requests require authentication using a bearer token:

Authorization: Bearer <token>

## Endpoints

### Model Management

#### List Models

GET /models

Returns a list of all registered models.

**Response**:

```json
[]
```

#### Get Model

GET /models/{model_id}

Returns details for a specific model.

**Response**:

```json
{}
```

#### Register Model

POST /models

Registers a new model.

**Request**:

```json
{}
```

**Response**:

```json
{}
```

### Dataset Management

#### List Datasets

GET /datasets

Returns a list of all registered datasets.

**Response**:

```json
[]
```

#### Get Dataset

GET /datasets/{dataset_id}

Returns details for a specific dataset.

**Response**:

```json
{}
```

#### Register Dataset

POST /datasets

Registers a new dataset.

**Request**:

```json
{}
```

**Response**:

```json
{}
```

### Evaluation

#### Evaluate Model

POST /evaluations

Evaluates a model on a dataset.

**Request**:

```json
{}
```

**Response**:

```json
{}
```

#### List Evaluations

GET /evaluations

Lists all evaluations, optionally filtered by model, dataset, or type.

**Query Parameters**:

- model_id: Filter by model ID
- dataset_id: Filter by dataset ID
- evaluation_type: Filter by evaluation type

**Response**:

```json
[]
```

### Benchmarking

#### Benchmark Model

POST /benchmarks

Benchmarks a model using a specific benchmark.

**Request**:

```json
{}
```

**Response**:

```json
{}
```

#### List Benchmarks

GET /benchmarks

Lists all available benchmarks.

**Response**:

```json
[]
```

#### Get Benchmark Results

POST /benchmarks/{benchmark_id}/results

Gets benchmark results for multiple models.

**Request**:

```json
{}
```

**Response**:

```json
[]
```

## Error Handling

The API uses standard HTTP status codes and returns error details in the response body:

```json
{}
```

Common status codes:

- 200 OK: Request succeeded
- 400 Bad Request: Invalid request
- 404 Not Found: Resource not found
- 500 Internal Server Error: Server error