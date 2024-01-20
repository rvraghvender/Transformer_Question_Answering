import os
import sys
from transformer_question_answering.logging.logger import logging
from transformer_question_answering.logging.exception import CustomException
from transformer_question_answering.entity import DataTransformationConfig
from transformer_question_answering.utils.common import get_question_and_facts, get_start_end_idx
from datasets import load_from_disk
from transformers import DistilBertTokenizerFast


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(config.tokenizer_path)

    def convert_examples_to_features(self, example):
        """Loads a data file into a list of `InputBatch`s."""
        encoding = self.tokenizer(example['sentences'], example['question'], truncation=True, padding=True, max_length=self.tokenizer.model_max_length)
        start_positions = encoding.char_to_token(example['str_idx'])
        end_positions = encoding.char_to_token(example['end_idx']-1)
        if start_positions is None:
            start_positions = self.tokenizer.model_max_length
        if end_positions is None:
            end_positions = self.tokenizer.model_max_length
        return {'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'start_positions': start_positions,
            'end_positions': end_positions}
    
    def convert(self):
        babi_dataset = load_from_disk(self.config.data_path)
        logging.info(f"Dataset loaded from disk path: {self.config.data_path}")
        flatten_babi_dataset = babi_dataset.flatten()
        processed_babi_dataset = flatten_babi_dataset.map(get_question_and_facts)
        processed_babi_dataset = processed_babi_dataset.map(get_start_end_idx)
        qa_dataset = processed_babi_dataset.map(self.convert_examples_to_features)
        qa_dataset = qa_dataset.remove_columns(['story.answer', 'story.id', 'story.supporting_ids', 'story.text', 'story.type'])
        qa_dataset.save_to_disk(os.path.join(self.config.root_dir, 'data'))