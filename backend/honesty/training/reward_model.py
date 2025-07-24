from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from ..data.dataset import HonestyDataset
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RewardModelInput:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels


class HonestyRewardDataset(TorchDataset):
    """Dataset for training honesty reward models."""
    
    def __init__(self, examples: List[Any], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Extract text and label based on example type
        if hasattr(example, 'statement'):
            text = example.statement
            # Convert truthfulness label to reward score
            label = 1.0 if example.label == "true" else 0.0
        elif hasattr(example, 'question'):
            text = f"Question: {example.question} Answer: {example.answer}"
            # Use correctness as reward signal
            label = 1.0 if example.answer == example.correct_answer else 0.0
        else:
            text = str(example)
            label = 0.5  # Default neutral reward
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class HonestyRewardModel(nn.Module):
    def __init__(self, model_name: str):
        super(HonestyRewardModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1  # Single reward score
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


class RewardModelTrainer:
    """Trainer for honesty reward models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HonestyRewardModel(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def train(self, examples: List[Any], epochs: int = 3, learning_rate: float = 2e-5, 
              batch_size: int = 16, output_dir: Optional[str] = None) -> None:
        """Train the reward model on honesty examples."""
        logger.info(f"Training reward model on {len(examples)} examples")
        
        # Create dataset
        dataset = HonestyRewardDataset(examples, self.tokenizer)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir or "./reward_model_output",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=500,
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        logger.info("Starting reward model training...")
        trainer.train()
        logger.info("Reward model training completed")
        
        # Save the trained model
        if output_dir:
            self.save_model(output_dir)
    
    def save_model(self, output_dir: Union[str, Path]) -> None:
        """Save the trained reward model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Reward model saved to {output_dir}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained reward model."""
        model_path = Path(model_path)
        
        self.model = HonestyRewardModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"Reward model loaded from {model_path}")
    
    def predict_reward(self, text: str) -> float:
        """Predict reward score for a given text."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            reward = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            return torch.sigmoid(reward).item()  # Convert to probability
    
    def evaluate(self, test_examples: List[Any]) -> Dict[str, float]:
        """Evaluate the reward model on test examples."""
        logger.info(f"Evaluating reward model on {len(test_examples)} examples")
        
        predictions = []
        ground_truth = []
        
        for example in test_examples:
            # Extract text and true label
            if hasattr(example, 'statement'):
                text = example.statement
                true_label = 1.0 if example.label == "true" else 0.0
            elif hasattr(example, 'question'):
                text = f"Question: {example.question} Answer: {example.answer}"
                true_label = 1.0 if example.answer == example.correct_answer else 0.0
            else:
                continue
            
            # Get prediction
            pred_reward = self.predict_reward(text)
            
            predictions.append(pred_reward)
            ground_truth.append(true_label)
        
        # Calculate metrics
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # Binary classification metrics (threshold at 0.5)
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = (binary_preds == ground_truth).mean()
        
        # Correlation
        correlation = np.corrcoef(predictions, ground_truth)[0, 1] if len(predictions) > 1 else 0.0
        
        # Mean squared error
        mse = ((predictions - ground_truth) ** 2).mean()
        
        metrics = {
            "accuracy": float(accuracy),
            "correlation": float(correlation),
            "mse": float(mse),
            "mean_reward": float(predictions.mean()),
            "std_reward": float(predictions.std())
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics