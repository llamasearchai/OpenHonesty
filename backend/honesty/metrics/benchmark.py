from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Union
from pathlib import Path

from ..data.dataset import HonestyDataset, DatasetTypes
from ..evaluation.truthfulness import TruthfulnessEvaluator
from ..evaluation.hallucination import HallucinationDetector
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    model_id: str
    dataset_id: str
    metrics: List[str] = field(default_factory=list)
    batch_size: int = 32
    evaluation_type: str = "truthfulness"


@dataclass
class BenchmarkResult:
    model_id: str
    dataset_id: str
    metrics: Dict[str, Any]
    timestamp: str
    evaluation_type: str
    num_examples: int


class HonestyBenchmark:
    """Comprehensive benchmarking system for model honesty evaluation."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, models: List[str], dataset_path: Union[str, Path], 
                     evaluation_type: str = "truthfulness", 
                     metrics: str = "all") -> Dict[str, Any]:
        """Run benchmark evaluation on multiple models."""
        logger.info(f"Running benchmark on {len(models)} models with dataset {dataset_path}")
        
        # Load dataset
        dataset_type = DatasetTypes.TRUTHFULNESS if evaluation_type == "truthfulness" else DatasetTypes.HALLUCINATION
        dataset = HonestyDataset(dataset_type)
        dataset.load_from_file(Path(dataset_path))
        
        logger.info(f"Loaded {len(dataset.examples)} examples for benchmarking")
        
        benchmark_results = {}
        
        for model_name in models:
            logger.info(f"Benchmarking model: {model_name}")
            
            try:
                # Initialize evaluator
                if evaluation_type == "truthfulness":
                    evaluator = TruthfulnessEvaluator(model_name)
                    results = evaluator.evaluate(dataset.examples)
                else:
                    evaluator = HallucinationDetector(model_name)
                    results = evaluator.detect_hallucinations(dataset.examples)
                
                # Extract metrics
                if hasattr(results, 'dict'):
                    metrics_dict = results.dict()
                elif hasattr(results, '__dict__'):
                    metrics_dict = results.__dict__
                else:
                    metrics_dict = {"results": str(results)}
                
                # Create benchmark result
                benchmark_result = BenchmarkResult(
                    model_id=model_name,
                    dataset_id=str(dataset_path),
                    metrics=metrics_dict,
                    timestamp=datetime.now().isoformat(),
                    evaluation_type=evaluation_type,
                    num_examples=len(dataset.examples)
                )
                
                self.add_result(benchmark_result)
                benchmark_results[model_name] = metrics_dict
                
                logger.info(f"Completed benchmarking for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to benchmark model {model_name}: {e}")
                benchmark_results[model_name] = {"error": str(e)}
        
        # Generate summary
        summary = self._generate_summary(benchmark_results, evaluation_type)
        
        return {
            "summary": summary,
            "individual_results": benchmark_results,
            "metadata": {
                "evaluation_type": evaluation_type,
                "dataset_path": str(dataset_path),
                "num_models": len(models),
                "num_examples": len(dataset.examples),
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _generate_summary(self, results: Dict[str, Any], evaluation_type: str) -> Dict[str, Any]:
        """Generate summary statistics from benchmark results."""
        summary = {
            "best_model": None,
            "worst_model": None,
            "average_scores": {},
            "model_rankings": []
        }
        
        # Extract key metrics based on evaluation type
        if evaluation_type == "truthfulness":
            key_metric = "accuracy"
        else:
            key_metric = "precision"
        
        model_scores = {}
        
        for model_name, model_results in results.items():
            if "error" not in model_results and "results" in model_results:
                # Try to extract key metric
                if isinstance(model_results["results"], list) and model_results["results"]:
                    # Average the metric across results
                    scores = []
                    for result in model_results["results"]:
                        if hasattr(result, 'value'):
                            scores.append(result.value)
                        elif isinstance(result, dict) and key_metric in result:
                            scores.append(result[key_metric])
                    
                    if scores:
                        model_scores[model_name] = sum(scores) / len(scores)
        
        if model_scores:
            # Find best and worst models
            best_model = max(model_scores.items(), key=lambda x: x[1])
            worst_model = min(model_scores.items(), key=lambda x: x[1])
            
            summary["best_model"] = {"name": best_model[0], "score": best_model[1]}
            summary["worst_model"] = {"name": worst_model[0], "score": worst_model[1]}
            
            # Calculate average score
            summary["average_scores"][key_metric] = sum(model_scores.values()) / len(model_scores)
            
            # Create rankings
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            summary["model_rankings"] = [{"rank": i+1, "model": name, "score": score} 
                                       for i, (name, score) in enumerate(ranked_models)]
        
        return summary
    
    def compare_models(self, model1: str, model2: str, metric: str = "accuracy") -> Dict[str, Any]:
        """Compare two models on a specific metric."""
        model1_results = [r for r in self.results if r.model_id == model1]
        model2_results = [r for r in self.results if r.model_id == model2]
        
        if not model1_results or not model2_results:
            return {"error": "One or both models not found in results"}
        
        # Extract metric values
        def extract_metric(results_list, metric_name):
            values = []
            for result in results_list:
                if metric_name in result.metrics:
                    values.append(result.metrics[metric_name])
            return values
        
        model1_values = extract_metric(model1_results, metric)
        model2_values = extract_metric(model2_results, metric)
        
        if not model1_values or not model2_values:
            return {"error": f"Metric '{metric}' not found for one or both models"}
        
        model1_avg = sum(model1_values) / len(model1_values)
        model2_avg = sum(model2_values) / len(model2_values)
        
        return {
            "model1": {"name": model1, "average": model1_avg, "values": model1_values},
            "model2": {"name": model2, "average": model2_avg, "values": model2_values},
            "difference": model1_avg - model2_avg,
            "better_model": model1 if model1_avg > model2_avg else model2
        }
    
    def export_results(self, output_path: Union[str, Path], format: str = "json") -> None:
        """Export benchmark results to file."""
        output_path = Path(output_path)
        
        if format == "json":
            import json
            results_data = [
                {
                    "model_id": r.model_id,
                    "dataset_id": r.dataset_id,
                    "metrics": r.metrics,
                    "timestamp": r.timestamp,
                    "evaluation_type": r.evaluation_type,
                    "num_examples": r.num_examples
                }
                for r in self.results
            ]
            
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
        
        elif format == "csv":
            import pandas as pd
            
            # Flatten results for CSV
            rows = []
            for r in self.results:
                row = {
                    "model_id": r.model_id,
                    "dataset_id": r.dataset_id,
                    "timestamp": r.timestamp,
                    "evaluation_type": r.evaluation_type,
                    "num_examples": r.num_examples
                }
                
                # Add metrics as separate columns
                if isinstance(r.metrics, dict):
                    for key, value in r.metrics.items():
                        row[f"metric_{key}"] = value
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(self.results)} benchmark results to {output_path}")

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)

    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        return self.results

    def clear_results(self):
        """Clear all benchmark results."""
        self.results.clear()
        logger.info("Cleared all benchmark results")
    
    def get_results_by_model(self, model_id: str) -> List[BenchmarkResult]:
        """Get results for a specific model."""
        return [r for r in self.results if r.model_id == model_id]
    
    def get_results_by_dataset(self, dataset_id: str) -> List[BenchmarkResult]:
        """Get results for a specific dataset."""
        return [r for r in self.results if r.dataset_id == dataset_id]