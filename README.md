# Text_Summarization_Infosys_Internship_Oct2024
OVERVIEW:
  This project focuses on implementing Text Summarization techniques using Natural Language Processing (NLP). Text summarization is the process of automatically 
  creating a concise and coherent summary of a larger text document while retaining its key information. This project explores three main categories:

  1. Extractive Methods
  2. Abstractive Methods
  3. Large Language Models (LLMs)

OBJECTIVE:

    • To implement and compare various extractive, abstractive, and LLM-based summarization techniques.
  
    • To evaluate the effectiveness of each method using standard metrics like ROUGE and BLEU scores.

FEATURES:

  Extractive Methods:
  
    Extractive summarization selects important sentences directly from the original text. The methods implemented include:
    
    - Frequency: Identifies and selects sentences containing frequently occurring words.
    
    - Latent Semantic Analysis (LSA): Uses singular value decomposition (SVD) to extract significant sentences.
    
    - TextRank: A graph-based ranking model inspired by Google's PageRank algorithm.
    
    - LexRank: Similar to TextRank, but incorporates cosine similarity for sentence ranking.
    
    - Luhn: Identifies sentences with high keyword density.

  Abstractive Methods:
  
    Abstractive summarization generates new sentences based on the meaning of the input text, resembling human-written summaries. The methods implemented include:
    
    - T5 (Text-to-Text Transfer Transformer)
    
    - BART (Bidirectional and Auto-Regressive Transformers)

  Large Language Models (LLMs):
  
    This project also integrates LLMs for summarization using advanced techniques:
    
    - Basic Summarization: Directly uses LLM APIs for summarizing text.
    
    - Map-Reduce: Breaks text into smaller chunks, summarizes each, and combines the results.
    
    - Iterative Refinement: Continuously refines summaries for better coherence and accuracy.

TECHNOLOGIES USED:
  - Python
  - Libraries: NLTK, Transformers, Sumy
  - Models: T5, BART, Open AI APIs
  - Tools: Google Colab

METHODOLOGY:

  1. Data Preparation:
     
      Input: Text data collected from publicly available sources like news articles, research papers, and blogs.
      Preprocessing: Tokenization, stopword removal, and text normalization.

  2. Implementation:
     
      Developed scripts for each summarization technique using Python.
      Leveraged pre-trained models for abstractive and LLM-based methods.
     
  3. Evaluation:

      Used metrics such as ROUGE (Recall-Oriented Understudy for Gisting Evaluation) and BLEU (Bilingual Evaluation Understudy) to measure summarization quality.

USE CASES:

  • Summarizing News Articles: Generate concise summaries for daily news consumption.
  
  • Legal Document Analysis: Quickly extract the essence of lengthy legal texts.
  
  • Research Paper Summarization: Save time by reviewing concise summaries of scientific papers.
  
  • Customer Support: Summarize chat logs to resolve queries faster.






