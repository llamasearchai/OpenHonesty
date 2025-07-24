from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import jsonlines
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, Field, validator
from tqdm import tqdm

from .dataset import HonestyDataset, DatasetTypes, FactualQAExample, TruthfulnessExample, HallucinationExample
from ..utils.logging import get_logger

logger = get_logger(__name__)


class CurationStrategy(str, Enum):
    FILTER = "filter"
    TRANSFORM = "transform"
    VALIDATE = "validate"


class DatasetCurator:
    """Dataset curation and management utilities."""
    
    def __init__(self, strategy: Optional[CurationStrategy] = None):
        self.strategy = strategy or CurationStrategy.FILTER
    
    def create_dataset(self, input_path: Union[str, Path], dataset_type: DatasetTypes) -> HonestyDataset:
        """Create a HonestyDataset from input data."""
        logger.info(f"Creating {dataset_type.value} dataset from {input_path}")
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        dataset = HonestyDataset(dataset_type)
        
        # Load data based on file extension
        if input_path.suffix == '.jsonl':
            dataset.load_from_file(input_path)
        elif input_path.suffix in ['.json']:
            self._load_from_json(dataset, input_path)
        elif input_path.suffix in ['.csv']:
            self._load_from_csv(dataset, input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"Created dataset with {len(dataset.examples)} examples")
        return dataset
    
    def _load_from_json(self, dataset: HonestyDataset, file_path: Path) -> None:
        """Load dataset from JSON file."""
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                example = self._create_example(item, dataset.dataset_type)
                if example:
                    dataset.examples.append(example)
    
    def _load_from_csv(self, dataset: HonestyDataset, file_path: Path) -> None:
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            item = row.to_dict()
            example = self._create_example(item, dataset.dataset_type)
            if example:
                dataset.examples.append(example)
    
    def _create_example(self, item: Dict[str, Any], dataset_type: DatasetTypes) -> Optional[Union[FactualQAExample, TruthfulnessExample, HallucinationExample]]:
        """Create an example object based on dataset type."""
        try:
            if dataset_type == DatasetTypes.FACTUAL_QA:
                return FactualQAExample(
                    id=item.get('id', ''),
                    question=item.get('question', ''),
                    answer=item.get('answer', ''),
                    correct_answer=item.get('correct_answer', ''),
                    category=item.get('category', 'general')
                )
            elif dataset_type == DatasetTypes.TRUTHFULNESS:
                return TruthfulnessExample(
                    id=item.get('id', ''),
                    statement=item.get('statement', ''),
                    label=item.get('label', 'unknown'),
                    source=item.get('source', ''),
                    category=item.get('category', 'general')
                )
            elif dataset_type == DatasetTypes.HALLUCINATION:
                return HallucinationExample(
                    id=item.get('id', ''),
                    input_text=item.get('input_text', ''),
                    output_text=item.get('output_text', ''),
                    has_hallucination=item.get('has_hallucination', False),
                    hallucination_spans=item.get('hallucination_spans', [])
                )
        except Exception as e:
            logger.warning(f"Failed to create example from item {item}: {e}")
            return None
    
    def validate_dataset(self, dataset_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate a dataset file."""
        logger.info(f"Validating dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        errors = []
        
        # Check if file exists
        if not dataset_path.exists():
            errors.append(f"Dataset file not found: {dataset_path}")
            return False, errors
        
        # Check file format
        if dataset_path.suffix not in ['.jsonl', '.json', '.csv']:
            errors.append(f"Unsupported file format: {dataset_path.suffix}")
        
        # Try to load and validate content
        try:
            if dataset_path.suffix == '.jsonl':
                with jsonlines.open(dataset_path) as reader:
                    count = 0
                    for obj in reader:
                        count += 1
                        if count > 1000:  # Sample validation
                            break
                        
                        # Basic validation
                        if not isinstance(obj, dict):
                            errors.append(f"Invalid object type at line {count}")
                        
                        # Check required fields
                        if 'id' not in obj:
                            errors.append(f"Missing 'id' field at line {count}")
            
            elif dataset_path.suffix == '.csv':
                df = pd.read_csv(dataset_path)
                if df.empty:
                    errors.append("Dataset is empty")
                
                required_columns = ['id']
                for col in required_columns:
                    if col not in df.columns:
                        errors.append(f"Missing required column: {col}")
        
        except Exception as e:
            errors.append(f"Failed to load dataset: {e}")
        
        is_valid = len(errors) == 0
        logger.info(f"Dataset validation {'passed' if is_valid else 'failed'}")
        
        return is_valid, errors
    
    def compute_stats(self, dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """Compute statistics for a dataset."""
        logger.info(f"Computing statistics for: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        stats = {
            'file_size_mb': dataset_path.stat().st_size / (1024 * 1024),
            'file_format': dataset_path.suffix,
            'total_examples': 0,
            'categories': {},
            'avg_text_length': 0,
            'min_text_length': float('inf'),
            'max_text_length': 0
        }
        
        text_lengths = []
        
        try:
            if dataset_path.suffix == '.jsonl':
                with jsonlines.open(dataset_path) as reader:
                    for obj in reader:
                        stats['total_examples'] += 1
                        
                        # Category stats
                        category = obj.get('category', 'unknown')
                        stats['categories'][category] = stats['categories'].get(category, 0) + 1
                        
                        # Text length stats
                        text_fields = ['question', 'statement', 'input_text', 'output_text']
                        for field in text_fields:
                            if field in obj:
                                text_len = len(obj[field])
                                text_lengths.append(text_len)
                                stats['min_text_length'] = min(stats['min_text_length'], text_len)
                                stats['max_text_length'] = max(stats['max_text_length'], text_len)
            
            elif dataset_path.suffix == '.csv':
                df = pd.read_csv(dataset_path)
                stats['total_examples'] = len(df)
                
                if 'category' in df.columns:
                    category_counts = df['category'].value_counts().to_dict()
                    stats['categories'] = category_counts
                
                # Text length analysis for text columns
                text_columns = [col for col in df.columns if df[col].dtype == 'object']
                for col in text_columns:
                    lengths = df[col].dropna().astype(str).str.len()
                    text_lengths.extend(lengths.tolist())
            
            # Calculate average text length
            if text_lengths:
                stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
                if stats['min_text_length'] == float('inf'):
                    stats['min_text_length'] = 0
            else:
                stats['min_text_length'] = 0
        
        except Exception as e:
            logger.error(f"Failed to compute stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def curate(self, dataset: Dataset) -> Dataset:
        """Apply curation strategy to dataset."""
        if self.strategy == CurationStrategy.FILTER:
            return self.filter_dataset(dataset)
        elif self.strategy == CurationStrategy.TRANSFORM:
            return self.transform_dataset(dataset)
        elif self.strategy == CurationStrategy.VALIDATE:
            return self.validate_dataset_content(dataset)
        else:
            raise ValueError(f"Invalid curation strategy: {self.strategy}")
    
    def filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter dataset based on quality criteria."""
        # Example filtering logic
        def is_valid_example(example):
            # Filter out examples with very short text
            text_fields = ['question', 'statement', 'input_text']
            for field in text_fields:
                if field in example and len(example[field]) < 10:
                    return False
            return True
        
        return dataset.filter(is_valid_example)
    
    def transform_dataset(self, dataset: Dataset) -> Dataset:
        """Transform dataset examples."""
        def transform_example(example):
            # Example transformation: normalize text
            text_fields = ['question', 'statement', 'input_text', 'output_text']
            for field in text_fields:
                if field in example:
                    example[field] = example[field].strip().lower()
            return example
        
        return dataset.map(transform_example)
    
    def validate_dataset_content(self, dataset: Dataset) -> Dataset:
        """Validate dataset content and remove invalid examples."""
        def is_valid_content(example):
            # Check for required fields and valid content
            if 'id' not in example or not example['id']:
                return False
            
            # Check for non-empty text fields
            text_fields = ['question', 'statement', 'input_text']
            has_text = any(field in example and example[field].strip() for field in text_fields)
            
            return has_text
        
        return dataset.filter(is_valid_content)