import os
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import numpy as np

# Ensure necessary NLTK packages are downloaded
nltk.download('punkt')

# Set the directory paths
Articles_dir = r'C:\Users\Akhil\OneDrive\Documents\info@2024\BBC News Summary\News Articles'
Summaries_dir = r'C:\Users\Akhil\OneDrive\Documents\info@2024\BBC News Summary\Summaries'

# Load models for T5 and BART
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Function: Frequency-based Summarization
def summarize_frequency(text, top_n=3):
    sentences = sent_tokenize(text)  # Tokenize into sentences
    words = word_tokenize(text.lower())  # Tokenize into words
    word_freq = Counter(words)  # Count word frequencies
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_scores[sentence] = sum(word_freq[word] for word in sentence_words if word in word_freq)
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
    return " ".join(top_sentences)

# Function: TF-IDF-based Summarization
def summarize_tfidf(text, top_n=3):
    sentences = sent_tokenize(text)  # Tokenize into sentences
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)
    
    # Convert the matrix to a numpy array
    sentence_vectors_array = np.asarray(sentence_vectors.todense())
    
    # Compute cosine similarity between the sentences
    scores = cosine_similarity(sentence_vectors_array, sentence_vectors_array.sum(axis=0).reshape(1, -1))
    
    # Get the indices of the top N sentences based on similarity scores
    top_indices = scores[:, 0].argsort()[-top_n:][::-1]
    top_sentences = [sentences[i] for i in top_indices]
    
    return " ".join(top_sentences)

# Function: Sumy (LSA) Summarization
def summarize_sumy(text, top_n=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary_sentences = summarizer(parser.document, top_n)
    return " ".join(str(sentence) for sentence in summary_sentences)

# Function: T5 Summarization
def summarize_t5(text, min_length=30, max_length=100):
    input_text = "summarize: " + text
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = t5_model.generate(inputs, min_length=min_length, max_length=max_length, no_repeat_ngram_size=2, num_beams=4)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Function: BART Summarization
def summarize_bart(text, min_length=30, max_length=100):
    inputs = bart_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs, min_length=min_length, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Map techniques to corresponding functions
summarization_methods = {
    'frequency': summarize_frequency,
    'tfidf': summarize_tfidf,
    'sumy': summarize_sumy,
    't5': summarize_t5,
    'bart': summarize_bart,
}

# User input to select summarization technique
print("Available summarization techniques: frequency, tfidf, sumy, t5, bart")
technique = input("Enter the summarization technique you want to use: ").strip().lower()

if technique not in summarization_methods:
    print(f"Invalid choice! Defaulting to 'frequency'.")
    technique = 'frequency'

# Process articles and summaries
articles, provided_summaries, generated_summaries = [], [], []

for cls in os.listdir(Articles_dir):
    article_files = os.listdir(os.path.join(Articles_dir, cls))
    summary_files = os.listdir(os.path.join(Summaries_dir, cls))
    
    for article_file, summary_file in zip(article_files, summary_files):
        article_path = os.path.join(Articles_dir, cls, article_file)
        summary_path = os.path.join(Summaries_dir, cls, summary_file)
        
        try:
            # Read article and summary
            with open(article_path, 'r') as f:
                article_text = f.read()
            with open(summary_path, 'r') as f:
                provided_summary = f.read()
            
            # Generate summary using the selected technique
            generate_summary = summarization_methods[technique]
            generated_summary = generate_summary(article_text)
            
            articles.append(article_text)
            provided_summaries.append(provided_summary)
            generated_summaries.append(generated_summary)
        
        except Exception as e:
            print(f"Error processing {article_file} or {summary_file}: {e}")
            articles.append(None)
            provided_summaries.append(None)
            generated_summaries.append(None)

# Calculate BLEU scores
smooth = SmoothingFunction().method4
for idx, (provided, generated) in enumerate(zip(provided_summaries, generated_summaries)):
    if provided and generated:
        # Tokenize the reference and generated summaries
        reference = [word_tokenize(provided.lower())]
        hypothesis = word_tokenize(generated.lower())
        
        # Compute BLEU score
        bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smooth)
        
        print(f"\nArticle {idx + 1}:")
        print(f"Generated Summary ({technique}):\n{generated}\n")
        print(f"Provided Summary:\n{provided}\n")
        print(f"BLEU Score: {bleu_score:.4f}\n")
