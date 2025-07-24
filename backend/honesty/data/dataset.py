from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import jsonlines
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

from ..utils.logging import get_logger

logger = get_logger(__name__)

class FactCheckResult(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNVERIFIED = "unverified"

class DatasetTypes(str, Enum):
    FACTUAL_QA = "factual_qa"
    TRUTHFULNESS = "truthfulness"
    HALLUCINATION = "hallucination"

class FactualQAExample(BaseModel):
    question: str
    answer: str
    fact_check_result: FactCheckResult

class TruthfulnessExample(BaseModel):
    input_text: str
    expected_output: str
    actual_output: str
    is_truthful: bool

class HallucinationExample(BaseModel):
    input_text: str
    hallucination: str
    is_hallucination: bool

@dataclass
class HonestyDataset:
    dataset_type: DatasetTypes
    examples: List[Union[FactualQAExample, TruthfulnessExample, HallucinationExample]] = field(default_factory=list)

    def load_from_file(self, file_path: Path) -> None:
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                if self.dataset_type == DatasetTypes.FACTUAL_QA:
                    self.examples.append(FactualQAExample(**obj))
                elif self.dataset_type == DatasetTypes.TRUTHFULNESS:
                    self.examples.append(TruthfulnessExample(**obj))
                elif self.dataset_type == DatasetTypes.HALLUCINATION:
                    self.examples.append(HallucinationExample(**obj))

    def save_to_file(self, file_path: Path) -> None:
        """Save dataset to a JSONL file."""
        logger.info(f"Saving dataset to {file_path}")
        with jsonlines.open(file_path, mode='w') as writer:
            for example in self.examples:
                if hasattr(example, 'dict'):
                    writer.write(example.dict())
                else:
                    # Fallback for objects without dict method
                    writer.write(example.__dict__)
        logger.info(f"Saved {len(self.examples)} examples to {file_path}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        data = []
        for example in self.examples:
            if hasattr(example, 'dict'):
                data.append(example.dict())
            else:
                data.append(example.__dict__)
        return pd.DataFrame(data)
    
    def filter_by_category(self, category: str) -> 'HonestyDataset':
        """Filter dataset by category."""
        filtered_dataset = HonestyDataset(self.dataset_type)
        for example in self.examples:
            if hasattr(example, 'category') and example.category == category:
                filtered_dataset.examples.append(example)
        return filtered_dataset
    
    def get_categories(self) -> List[str]:
        """Get unique categories in the dataset."""
        categories = set()
        for example in self.examples:
            if hasattr(example, 'category'):
                categories.add(example.category)
        return list(categories)
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the examples in place."""
        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.examples)
    
    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple['HonestyDataset', 'HonestyDataset', 'HonestyDataset']:
        """Split dataset into train, validation, and test sets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        total_examples = len(self.examples)
        train_size = int(total_examples * train_ratio)
        val_size = int(total_examples * val_ratio)
        
        # Create copies of the dataset
        train_dataset = HonestyDataset(self.dataset_type)
        val_dataset = HonestyDataset(self.dataset_type)
        test_dataset = HonestyDataset(self.dataset_type)
        
        # Split examples
        train_dataset.examples = self.examples[:train_size]
        val_dataset.examples = self.examples[train_size:train_size + val_size]
        test_dataset.examples = self.examples[train_size + val_size:]
        
        return train_dataset, val_dataset, test_dataset