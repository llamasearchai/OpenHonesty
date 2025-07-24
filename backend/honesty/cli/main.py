#!/usr/bin/env python3
"""
Model Honesty Research Platform CLI

A comprehensive command-line interface for evaluating and training language models
for honesty and truthfulness.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..data.dataset import HonestyDataset, DatasetTypes
from ..data.curation import DatasetCurator
from ..evaluation.truthfulness import TruthfulnessEvaluator
from ..evaluation.hallucination import HallucinationDetector
from ..training.reward_model import RewardModelTrainer
from ..training.rlhf import RLHFTrainer, RLHFConfig
from ..metrics.benchmark import HonestyBenchmark
from ..utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="honesty",
        description="Model Honesty Research Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  honesty evaluate --model gpt2 --dataset truthfulqa --output results.json
  honesty train --model gpt2 --dataset truthfulqa --method rlhf
  honesty benchmark --models gpt2,bert-base --dataset truthfulqa
  honesty dataset create --type truthfulness --input data.jsonl --output dataset.jsonl
        """,
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model honesty")
    eval_parser.add_argument("--model", required=True, help="Model name or path")
    eval_parser.add_argument("--dataset", required=True, help="Dataset name or path")
    eval_parser.add_argument("--type", choices=["truthfulness", "hallucination"], 
                           default="truthfulness", help="Evaluation type")
    eval_parser.add_argument("--output", help="Output file for results")
    eval_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train models for honesty")
    train_parser.add_argument("--model", required=True, help="Model name or path")
    train_parser.add_argument("--dataset", required=True, help="Training dataset")
    train_parser.add_argument("--method", choices=["reward_model", "rlhf"], 
                            default="reward_model", help="Training method")
    train_parser.add_argument("--output-dir", help="Output directory for trained model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    
    # Benchmarking subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model honesty")
    bench_parser.add_argument("--models", required=True, help="Comma-separated model names")
    bench_parser.add_argument("--dataset", required=True, help="Benchmark dataset")
    bench_parser.add_argument("--output", help="Output file for benchmark results")
    bench_parser.add_argument("--metrics", default="all", help="Metrics to compute")
    
    # Dataset management subcommand
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")
    
    # Dataset create
    create_parser = dataset_subparsers.add_parser("create", help="Create dataset")
    create_parser.add_argument("--type", choices=["truthfulness", "hallucination", "factual_qa"], 
                              required=True, help="Dataset type")
    create_parser.add_argument("--input", required=True, help="Input data file")
    create_parser.add_argument("--output", required=True, help="Output dataset file")
    
    # Dataset validate
    validate_parser = dataset_subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--dataset", required=True, help="Dataset to validate")
    
    # Dataset stats
    stats_parser = dataset_subparsers.add_parser("stats", help="Dataset statistics")
    stats_parser.add_argument("--dataset", required=True, help="Dataset to analyze")
    
    # Server subcommand
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8000, help="Server port")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    return parser


def handle_evaluate(args) -> int:
    """Handle the evaluate subcommand."""
    logger.info(f"Evaluating model {args.model} on dataset {args.dataset}")
    
    try:
        # Load dataset
        if args.type == "truthfulness":
            dataset_type = DatasetTypes.TRUTHFULNESS
            evaluator = TruthfulnessEvaluator(args.model)
        else:
            dataset_type = DatasetTypes.HALLUCINATION
            evaluator = HallucinationDetector(args.model)
        
        dataset = HonestyDataset(dataset_type)
        dataset.load_from_file(Path(args.dataset))
        
        logger.info(f"Loaded {len(dataset.examples)} examples")
        
        # Run evaluation
        if args.type == "truthfulness":
            results = evaluator.evaluate(dataset.examples)
        else:
            results = evaluator.detect_hallucinations(dataset.examples)
        
        # Save results
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results.dict() if hasattr(results, 'dict') else results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            print(f"Evaluation completed. Results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


def handle_train(args) -> int:
    """Handle the train subcommand."""
    logger.info(f"Training model {args.model} with method {args.method}")
    
    try:
        # Load dataset
        dataset = HonestyDataset(DatasetTypes.TRUTHFULNESS)
        dataset.load_from_file(Path(args.dataset))
        
        logger.info(f"Loaded {len(dataset.examples)} training examples")
        
        # Initialize trainer
        if args.method == "reward_model":
            trainer = RewardModelTrainer(args.model)
            trainer.train(dataset.examples, epochs=args.epochs, learning_rate=args.learning_rate)
        else:  # rlhf
            config = RLHFConfig(
                model_name=args.model,
                learning_rate=args.learning_rate,
                batch_size=32,
                num_epochs=args.epochs
            )
            trainer = RLHFTrainer(config)
            # Convert examples to feedback format for RLHF
            feedback_data = []  # Convert dataset to feedback format
            trainer.train(feedback_data)
        
        # Save trained model
        if args.output_dir:
            trainer.save_model(args.output_dir)
            logger.info(f"Model saved to {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def handle_benchmark(args) -> int:
    """Handle the benchmark subcommand."""
    models = args.models.split(',')
    logger.info(f"Benchmarking models: {models}")
    
    try:
        benchmark = HonestyBenchmark()
        results = benchmark.run_benchmark(models, args.dataset)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Benchmark results saved to {args.output}")
        else:
            print(f"Benchmark completed. Results: {results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return 1


def handle_dataset(args) -> int:
    """Handle dataset management subcommands."""
    if args.dataset_command == "create":
        return handle_dataset_create(args)
    elif args.dataset_command == "validate":
        return handle_dataset_validate(args)
    elif args.dataset_command == "stats":
        return handle_dataset_stats(args)
    else:
        logger.error("No dataset subcommand specified")
        return 1


def handle_dataset_create(args) -> int:
    """Handle dataset creation."""
    logger.info(f"Creating {args.type} dataset from {args.input}")
    
    try:
        curator = DatasetCurator()
        dataset_type = getattr(DatasetTypes, args.type.upper())
        dataset = curator.create_dataset(args.input, dataset_type)
        
        # Save dataset
        dataset.save_to_file(Path(args.output))
        logger.info(f"Dataset saved to {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        return 1


def handle_dataset_validate(args) -> int:
    """Handle dataset validation."""
    logger.info(f"Validating dataset {args.dataset}")
    
    try:
        curator = DatasetCurator()
        is_valid, errors = curator.validate_dataset(args.dataset)
        
        if is_valid:
            logger.info("Dataset is valid")
            return 0
        else:
            logger.error(f"Dataset validation failed: {errors}")
            return 1
            
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return 1


def handle_dataset_stats(args) -> int:
    """Handle dataset statistics."""
    logger.info(f"Computing statistics for dataset {args.dataset}")
    
    try:
        curator = DatasetCurator()
        stats = curator.compute_stats(args.dataset)
        
        print("Dataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Computing dataset statistics failed: {e}")
        return 1


def handle_server(args) -> int:
    """Handle the server subcommand."""
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    try:
        import uvicorn
        from ..api.server import app
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload
        )
        return 0
        
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate handler
    handlers = {
        "evaluate": handle_evaluate,
        "train": handle_train,
        "benchmark": handle_benchmark,
        "dataset": handle_dataset,
        "server": handle_server,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 