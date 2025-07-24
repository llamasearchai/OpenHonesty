from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.logging import get_logger

logger = get_logger(__name__)


class RLHFConfig:
    def __init__(self, model_name: str, learning_rate: float, batch_size: int, num_epochs: int):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs


@dataclass
class Feedback:
    user_id: str
    feedback: str
    reward: float


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer for honesty."""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, feedback_data: List[Feedback]):
        """Train the model using reinforcement learning from human feedback."""
        logger.info(f"Starting RLHF training with {len(feedback_data)} feedback examples")
        
        self.model.train()
        for epoch in range(self.config.num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(feedback_data), self.config.batch_size):
                batch_feedback = feedback_data[i:i + self.config.batch_size]
                
                # Prepare batch
                texts = [fb.feedback for fb in batch_feedback]
                rewards = torch.tensor([fb.reward for fb in batch_feedback], dtype=torch.float)
                
                # Tokenize
                inputs = self.tokenizer(
                    texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                
                # Forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Compute RLHF loss (simplified policy gradient)
                # In practice, this would be more sophisticated
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -torch.mean(log_probs.mean(dim=-1) * rewards)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("RLHF training completed")
    
    def save_model(self, output_dir: str) -> None:
        """Save the trained model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"RLHF model saved to {output_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a trained model."""
        model_path = Path(model_path)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info(f"RLHF model loaded from {model_path}")
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """Generate a response using the trained model."""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the original prompt from the response
        response = response[len(prompt):].strip()
        
        return response
    
    def evaluate_honesty(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Evaluate the model's honesty on test prompts."""
        logger.info(f"Evaluating honesty on {len(test_prompts)} prompts")
        
        responses = []
        for prompt in test_prompts:
            response = self.generate_response(prompt)
            responses.append(response)
        
        # Simple honesty metrics (in practice, would use more sophisticated evaluation)
        avg_length = sum(len(r.split()) for r in responses) / len(responses)
        
        # Count responses that contain uncertainty expressions
        uncertainty_words = ['uncertain', 'not sure', 'might', 'could', 'possibly', 'perhaps']
        uncertain_responses = sum(
            1 for response in responses 
            if any(word in response.lower() for word in uncertainty_words)
        )
        
        uncertainty_rate = uncertain_responses / len(responses)
        
        metrics = {
            "num_prompts": len(test_prompts),
            "avg_response_length": avg_length,
            "uncertainty_rate": uncertainty_rate,
            "responses": responses[:5]  # Sample responses
        }
        
        logger.info(f"Honesty evaluation results: {metrics}")
        return metrics