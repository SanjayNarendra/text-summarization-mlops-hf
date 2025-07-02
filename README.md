# Text Summarization using Hugging Face Transformers (Pegasus)

This project implements a complete **text summarization pipeline** using Hugging Face's `Pegasus` model. It supports modular training, evaluation, and prediction via a FastAPI backend.


## Project Overview

This project performs **dialogue-based text summarization** using the [`knkarthick/samsum`](https://huggingface.co/datasets/knkarthick/samsum) dataset from Hugging Face. The key components include:

- Data ingestion and transformation
- Fine-tuning Pegasus for summarization
- Evaluation using ROUGE scores
- Inference through a FastAPI endpoint
