# Text Summarization Project using NLP

This repository focuses on text summarization using various Natural Language Processing (NLP) techniques. The project implements both **extractive** and **abstractive** summarization models, evaluates their performance using ROUGE and BLEU scores, and provides an interactive Gradio interface for users to test the models.

## Features

### Summarization Models

#### **Extractive Models**
- **Frequency-based Summarization**: Extracts key sentences based on word frequency.
- **TF-IDF Summarization**: Utilizes TF-IDF scores to identify the most relevant sentences.
- **Sumy (LSA)**: Employs Latent Semantic Analysis for summarization.
- **LexRank**: A graph-based ranking model for sentence extraction (implemented using the `sumy` library).

#### **Abstractive Models**
- **T5**: A transformer model that generates summaries by rephrasing and restructuring content.
- **BART**: Another transformer model designed for both text generation and summarization.
- **LLM (Large Language Models)**: Summarization using external APIs or pre-trained large-scale generative models.

### Evaluation Metrics
- **ROUGE**: Compares the n-grams of the generated summaries with reference summaries to measure recall, precision, and F1-score.
- **BLEU**: Measures the quality of generated text by comparing n-grams with reference summaries, focusing on precision.

### Interactive Gradio Application
- Input custom text or upload files.
- Choose a summarization technique from extractive or abstractive models.
- Generate summaries and view evaluation scores in real time.


