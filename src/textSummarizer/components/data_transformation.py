import os
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict

from src.textSummarizer.config.configuration import DataTransformationConfig


class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def convert_examples_to_features(self, example_batch):
        dialogues = [d if isinstance(d, str) else "" for d in example_batch['dialogue']]
        summaries = [s if isinstance(s, str) else "" for s in example_batch['summary']]

        input_encodings = self.tokenizer(dialogues, max_length = 1024, truncation = True )

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(summaries, max_length = 128, truncation = True )

        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)

        # Apply conversion split-by-split
        dataset_samsum_pt = {}
        for split in dataset_samsum.keys():
            logger.info(f"Processing split: {split}")
            dataset_samsum_pt[split] = dataset_samsum[split].map(
                self.convert_examples_to_features,
                batched=True
            )

        dataset_samsum_pt = DatasetDict(dataset_samsum_pt)
        
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))