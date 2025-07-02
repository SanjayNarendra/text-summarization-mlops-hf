import os
from datasets import load_dataset
from src.textSummarizer.logging import logger

from src.textSummarizer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def fetch_and_save_dataset(self):
        logger.info("Loading dataset from Hugging Face...")
        dataset = load_dataset(self.config.dataset_name)

        logger.info("Saving raw dataset locally...")
        dataset.save_to_disk(self.config.raw_dataset_dir)
        
        logger.info("Saving dataset splits locally...")
        dataset['train'].to_csv(os.path.join(self.config.root_dir, "samsum-train.csv"), index=False)
        dataset['test'].to_csv(os.path.join(self.config.root_dir, "samsum-test.csv"), index=False)
        dataset['validation'].to_csv(os.path.join(self.config.root_dir, "samsum-validation.csv"), index=False)

        logger.info("Data ingestion from Hugging Face completed.")
