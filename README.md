
# Evaluation of Fine-Tuning Llama-2 for Domain-specific Question Answering


## Llama-2 QnA

Llama-2 QnA is a project aimed at fine-tuning the Llama-2 model by Meta for domain-specific question answering. This repository contains code, data, and resources required to adapt and fine-tune the model for answering questions related to specific knowledge domains, such as physics and scientific literature.

## Abstract

Adapting Large Language Models (LLMs) for downstream tasks involves understanding the new knowledge domain and fine-tuning the model for specific tasks. This project employs the Llama-2 model with 7B parameters and fine-tunes it for question-answering tasks using two different datasets: a collection of physics books and a scientific paper. The models successfully learned the new knowledge domains and provided representative answers to queries.

## Project Structure

### Directories and Notable Files

- **llama2_conda_deps.yml**: Contains Conda environment dependencies.
- **data**: Stores data files used for training and evaluation.
  - Various CSV files for different training datasets.
- **figs**: Stores figures and visualizations.
  - Various figures for training loss and evaluation metrics.
- **notebooks**: Jupyter notebooks for experimentation and development.
- **paper_13b**: Resources related to the 13b version of the Paper model.
  - `finetuning_qna_paper`
  - `pretraining_paper`
- **paper_7b**: Resources related to the 7b version of the Paper model.
  - `finetuning_qna_paper`
  - `pretraining_paper`
- **physics_mk_books_7b**: Resources related to the 7b version of the PhysX model.
  - `finetuning_qna`
  - `pretraining_books`
- **install_gpu_driver.py**: Script used within the Google Cloud Platform (GCP) for installing necessary GPU drivers (NVIDIA L4 GPU).
    - [original file](https://github.com/GoogleCloudPlatform/compute-gpu-installation/blob/main/linux/install_gpu_driver.py)


## Features

- **Domain-Specific Fine-Tuning**: Fine-tunes the Llama-2 model using domain-specific datasets, such as physics books and scientific papers.
- **Parameter-Efficient Training**: Utilizes QLoRA for memory-efficient fine-tuning with 4-bit quantization, significantly reducing the number of trainable parameters.
- **Question & Answer Generation**: Generates high-quality Q&A pairs from domain-specific texts using Llama2-7b-chat.
- **Text Processing**: Efficiently extracts and processes text data from various sources, including books and papers.
- **Model Evaluation**: Implements rigorous evaluation metrics like Perplexity (PPL) and Rouge to assess model performance.
- **Scalable Architecture**: Designed to handle multiple queries simultaneously, making it suitable for real-time applications.
- **Comprehensive Documentation**: Includes detailed notebooks for data processing, model training, and evaluation to facilitate easy replication and experimentation.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/aRibra/llama_qna.git
    cd llama_qna
    ```

2. **Set up the environment**:
    Use the Conda environment dependencies to set up your Conda environment.

    ```bash
    conda env create -f llama2_conda_deps.yml
    conda activate llama2
    ```

3. **Install GPU drivers** (for GPU):
    ```bash
    python install_gpu_driver.py
    ```

## Usage

Run the main script or Jupyter notebooks in the `notebooks` directory to start the QnA system and experiment with the model.

### Example Notebooks

- `Physics_book_loader.ipynb`: Load and preprocess physics book data.
- `QnA_GPT_Generator-coaxnn-paper.ipynb`: Generate QnA pairs using the CoAxNN paper.
- `QnA_PhysX_Generator.ipynb`: Generate QnA pairs for physics data.
- `arxiv_loader.ipynb`: Load data from arXiv.
- `coaxnn_paper_loader.ipynb`: Load and preprocess CoAxNN paper data.

## Methodology

### Datasets

- **Physics Books**: Text extracted from books by Michio Kaku, including "Hyperspace," "Physics of the Impossible," "Physics of the Future," and "The Future of the Mind."
- **Scientific Paper**: Text from a recent paper on pruning neural networks.

### Training Process

1. **Text Extraction**: Using PDFMinerLoader and RecursiveCharacterTextSplitter to extract and split text into manageable chunks.
2. **Q&A Generation**: Using Llama2-7b-chat to generate Q&A pairs from text chunks.
3. **Fine-Tuning**: Using QLoRA for memory-efficient fine-tuning in 4-bit quantization.

### Evaluation Metrics

- **Perplexity (PPL)**: Measures model performance on pre-training datasets.
- **Rouge**: Evaluates the quality of generated answers against reference answers.

## Results

The fine-tuned models showed significant improvements in domain-specific tasks, with reduced perplexity and high Rouge scores indicating effective adaptation to new knowledge domains.

### Evaluation Metrics

- **Perplexity (PPL)**: A lower perplexity value indicates better model performance. The fine-tuned models demonstrated significant improvements in perplexity compared to the original Llama-2 model.
  - **MK-PhysX-7b-PT**: Reduced PPL from 6.56 to 2.47 on the MK-PhysX-text dataset.
  - **Paper-7b-PT**: Reduced PPL from 6.91 to 2.49 on the Paper-text dataset.

- **Rouge Scores**: High Rouge scores indicate better quality and relevance of generated answers.
  - **MK-PhysX-7b-qna**: Achieved 40.57% Rouge-1, 26.15% Rouge-2, and 37.50% Rouge-L at checkpoint 2600.
  - **Paper-7b-qna**: Achieved 70.78% Rouge-1, 62.56% Rouge-2, and 69.20% Rouge-L at checkpoint 360.

### Training Loss

- **MK-PhysX-7b-PT**:
  - Pre-training (text): Demonstrated a steady decrease in training loss across epochs.
  - Fine-tuning (Q&A): Further reduced training loss, indicating effective adaptation to the Q&A task.
- **Paper-7b-PT**:
  - Pre-training (text): Showed consistent reduction in training loss.
  - Fine-tuning (Q&A): Achieved lower training loss, confirming successful fine-tuning for the Q&A task.

### Sample Q&A Responses

#### Physics Books Model: MK-PhysX-7b-qna
- **Prompt**: How do the apes treat humans in the advanced civilization?
  - **Response**: According to the text, the advanced apes treat humans as pets or servants.

- **Prompt**: Can time travel create problems in the timeline?
  - **Response**: Yes, time travel can create problems in the timeline. In fact, it is a common feature of many stories about time travel, where the time traveler inadvertently alters the course of history in unexpected ways. For example, if you went back in time to witness the Battle of Gettysburg, you might inadvertently tip the outcome of the battle, changing the course of history forever. Similarly, if you were to go back in time to assassinate Abraham Lincoln, you would alter the course of the Civil War and ultimately the entire course of American history.
  
#### Scientific Paper Model: Paper-7b-qna
- **Prompt**: What is proposed in the CoAxNN paper?
  - **Response**: In the CoAxNN paper, it is proposed to combine the staging-based and pruning-based approximate strategies to achieve efficient model optimization.

- **Prompt**: What does CoAxNN stand for?
  - **Response**: CoAxNN stands for Conditional Approximate Neural Networks.



---
