# Text_Summarization_Infosys_Internship_Oct2024
This Text Summarization Project aims to develop a tool that efficiently condenses long articles, papers, or documents into concise summaries while preserving the key information and context. Utilizing advanced Natural Language Processing (NLP) techniques, this project focuses on both extractive and abstractive summarization methods.


### Summarization Techniques
  We start off with the understanding of Text Summarization and the ways it can be done in:
  1. Extractive Techniques
  2. Abstractive Techniques

#### Implementation of Extractive Summarization Techniques
  Text Summarization using Extractive Summarization Techniques:
  1. The frequency method
  2. Sumy
  3. Lex Rank
  4. LSA
     
  Later stage of the notebook consists of the combined code to implement the above algorithms, following which we compare the four models using ROUGE score.
  
  Methodology of comparision:
  
  1. Iterate Over the Dataset: For each entry in the dataset, extract the "article" as the input text and the "highlights" as the reference summary.

  2. Generate Summaries Using Each Model: For each article, generate a summary using all four models (Frequency Method, Sumy, Lex Rank, LSA).

  3. Calculate ROUGE Scores for Each Model: Use the "highlights" as the reference summary and compute ROUGE scores for each generated summary using the rouge_score library. This will involve computing metrics like     ROUGE-1, ROUGE-2, and ROUGE-L.

  4. Store Results: Store the ROUGE scores for each article and model.

  5. Aggregate Scores: After iterating through the dataset, calculate the average ROUGE scores for each model across all articles. This will give you a comparative measure of how well each model performs.

  Open file Extractive_Summarization_Techniques.ipynb and run all the cells to demonstrate the working of each algorithm.

