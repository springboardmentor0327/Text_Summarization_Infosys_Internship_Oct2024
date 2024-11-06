import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import nltk

# Ensure punkt tokenizer is downloaded
nltk.download('punkt')

# Set the directory path to the articles
Articles_dir = r'C:\Users\Akhil\Downloads\BBC News Summary\News Articles'

# Get all the subdirectories (categories) of articles
classes = os.listdir(Articles_dir)

# Initialize empty lists to store articles and their generated summaries
articles = []
summaries_1 = []  # First generated summaries
summaries_2 = []  # Second generated summaries  

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')  # Use CPU (change to 'cuda' for GPU)

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to generate a summary using T5 model
def generate_summary(text, num_beams=10):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text, num_beams=num_beams, no_repeat_ngram_size=3, min_length=30, max_length=100, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function to calculate BLEU score between the reference (article) and the generated summary
def calculate_bleu(reference, hypothesis):
    # Tokenize the reference and hypothesis into words
    reference_tokens = [word_tokenize(reference.lower())]  # List of tokenized words for reference
    hypothesis_tokens = word_tokenize(hypothesis.lower())  # Tokenized hypothesis (generated summary)
    
    # corpus_bleu expects a list of references (each reference is a list of tokens)
    return corpus_bleu([reference_tokens], [hypothesis_tokens])

# Iterate over each class and file in the articles directory
for cls in classes:
    files = os.listdir(os.path.join(Articles_dir, cls))
    for file in files:
        article_file_path = os.path.join(Articles_dir, cls, file)
        
        try:
            # Read the article text from the file
            with open(article_file_path, 'r') as f:
                article_text = '.'.join([line.rstrip() for line in f.readlines()])
                articles.append(article_text)
            
            # Generate the first summary for the article
            print(f"Generating first summary for {file}...")
            generated_summary_1 = generate_summary(article_text, num_beams=10)
            summaries_1.append(generated_summary_1)
            
            # Generate the second summary for the article (using a different number of beams)
            print(f"Generating second summary for {file}...")
            generated_summary_2 = generate_summary(article_text, num_beams=5)  # Using fewer beams for variation
            summaries_2.append(generated_summary_2)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            articles.append(None)
            summaries_1.append(None)
            summaries_2.append(None)

# Ensure both articles and summaries have the same length
if len(articles) != len(summaries_1) or len(articles) != len(summaries_2):
    difference = len(articles) - len(summaries_1)
    summaries_1.extend([None] * difference)
    summaries_2.extend([None] * difference)

# Create a DataFrame to store articles and their summaries
dataset = pd.DataFrame({'Articles': articles, 'Summary 1': summaries_1, 'Summary 2': summaries_2})

# Evaluate the generated summaries using ROUGE and BLEU scores
for idx, row in dataset.iterrows():
    article = row['Articles']
    generated_summary_1 = row['Summary 1']
    generated_summary_2 = row['Summary 2']
    
    if article is not None and generated_summary_1 is not None and generated_summary_2 is not None:
        # Calculate ROUGE scores for the first summary
        scores_1 = scorer.score(article, generated_summary_1)
        
        # Calculate BLEU score between the two generated summaries
        bleu_score = calculate_bleu(generated_summary_1, generated_summary_2)
        
        print(f"\nArticle {idx + 1}:\n{article}\n")
        print(f"Generated Summary 1 {idx + 1}:\n{generated_summary_1}\n")
        
        # Print ROUGE Scores for the first summary only
        print(f"Article {idx + 1} ROUGE Scores for Summary 1:")
        print(f"ROUGE-1: {scores_1['rouge1']}")
        print(f"ROUGE-2: {scores_1['rouge2']}")
        print(f"ROUGE-L: {scores_1['rougeL']}\n")
        
        # Print BLEU score between the two summaries
        print(f"Article {idx + 1} BLEU Score between the two summaries: {bleu_score}\n")
    else:
        print(f"Article {idx + 1}: Error processing, could not generate summaries or no article found.\n")
