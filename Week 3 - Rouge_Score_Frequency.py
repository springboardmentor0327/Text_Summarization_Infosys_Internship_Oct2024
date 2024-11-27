import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from rouge_score import rouge_scorer

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')

# Set the directory paths
Articles_dir = r'C:\Users\Akhil\OneDrive\Documents\info@2024\BBC News Summary\News Articles'
Summaries_dir = r'C:\Users\Akhil\OneDrive\Documents\info@2024\BBC News Summary\Summaries'

# Get all the subdirectories (categories) of articles and summaries
article_classes = os.listdir(Articles_dir)
summary_classes = os.listdir(Summaries_dir)

# Initialize empty lists to store articles, provided summaries, and generated summaries
articles = []
provided_summaries = []
generated_summaries = []

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to generate a summary using the frequency method
def generate_summary_frequency(text, top_n=3):
    """
    Summarizes the input text by selecting the top N sentences with the most frequent words.
    
    Parameters:
    - text: The input text to summarize.
    - top_n: The number of sentences to include in the summary.
    
    Returns:
    - summary: A string containing the summarized text.
    """
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    words = word_tokenize(text.lower())  # Tokenize text into words
    word_freq = Counter(words)  # Count the frequency of each word
    
    # Score each sentence based on the sum of the frequencies of its words
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(word_freq[word] for word in sentence_words if word in word_freq)
    
    # Sort sentences by score and select the top N
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
    summary = " ".join(top_sentences)
    
    return summary

# Iterate over each class and file in the articles and summaries directories
for cls in article_classes:
    article_files = os.listdir(os.path.join(Articles_dir, cls))
    summary_files = os.listdir(os.path.join(Summaries_dir, cls))
    
    for article_file, summary_file in zip(article_files, summary_files):
        article_file_path = os.path.join(Articles_dir, cls, article_file)
        summary_file_path = os.path.join(Summaries_dir, cls, summary_file)
        
        try:
            # Read the article text
            with open(article_file_path, 'r') as f:
                article_text = '.'.join([line.rstrip() for line in f.readlines()])
                articles.append(article_text)
            
            # Generate summary for the article using frequency method
            print(f"Generating summary for {article_file}...")
            generated_summary = generate_summary_frequency(article_text, top_n=3)
            generated_summaries.append(generated_summary)
            
            # Read the provided summary text
            with open(summary_file_path, 'r') as f:
                provided_summary_text = '.'.join([line.rstrip() for line in f.readlines()])
                provided_summaries.append(provided_summary_text)

        except Exception as e:
            print(f"Error processing files {article_file} or {summary_file}: {e}")
            articles.append(None)
            provided_summaries.append(None)
            generated_summaries.append(None)

# Ensure all lists are of the same length
max_len = max(len(articles), len(provided_summaries), len(generated_summaries))
articles.extend([None] * (max_len - len(articles)))
provided_summaries.extend([None] * (max_len - len(provided_summaries)))
generated_summaries.extend([None] * (max_len - len(generated_summaries)))

# Create a DataFrame to store articles, provided summaries, and generated summaries
dataset = pd.DataFrame({
    'Articles': articles,
    'Provided Summaries': provided_summaries,
    'Generated Summaries': generated_summaries
})

# Evaluate the generated summaries against the provided summaries using ROUGE scores
for idx, row in dataset.iterrows():
    provided_summary = row['Provided Summaries']
    generated_summary = row['Generated Summaries']
    
    if provided_summary is not None and generated_summary is not None:
        # Calculate ROUGE scores
        scores = scorer.score(provided_summary, generated_summary)
        
        print(f"\nArticle {idx + 1}:")
        print(f"Generated Summary:\n{generated_summary}\n")
        print(f"Provided Summary:\n{provided_summary}\n")
        
        # Print ROUGE Scores
        print(f"ROUGE Scores for Article {idx + 1}:")
        print(f"ROUGE-1: {scores['rouge1']}")
        print(f"ROUGE-2: {scores['rouge2']}")
        print(f"ROUGE-L: {scores['rougeL']}\n")
    else:
        print(f"Article {idx + 1}: Error processing, could not compare summaries.\n")
