# Importing required libraries
import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict
import evaluate
from nltk.tokenize import sent_tokenize
import warnings

warnings.filterwarnings('ignore')

# Define paths correctly
articles_path = 'C:/Users/Akhil/Downloads/BBC News Summary/News Articles/'
summaries_path = 'C:/Users/Akhil/Downloads/BBC News Summary/Summaries/'

# List directories in the articles path for debugging
print("Directories in articles path:")
for path in os.listdir(articles_path):
    print(path)

articles = []
summaries = []
file_arr = []

for path in os.listdir(articles_path):
    files = os.listdir(articles_path + path)
    for file in files:
        article_file_path = os.path.join(articles_path, path, file)
        summary_file_path = os.path.join(summaries_path, path, file)
        
        try:
            with open(article_file_path, 'r') as f:
                article = '.'.join([line.rstrip() for line in f.readlines()])
            
            with open(summary_file_path, 'r') as f:
                summary = '.'.join([line.rstrip() for line in f.readlines()])
            
            # Append only if both article and summary are present
            articles.append(article)
            summaries.append(summary)
            file_arr.append(os.path.join(path, file))
        
        except FileNotFoundError:
            print(f"File not found: {article_file_path} or {summary_file_path}")

# Check lengths before creating DataFrame
print(f"file_arr length: {len(file_arr)}")
print(f"articles length: {len(articles)}")
print(f"summaries length: {len(summaries)}")

if len(file_arr) == len(articles) == len(summaries):
    df = pd.DataFrame({'path': file_arr, 'article': articles, 'summary': summaries})
else:
    print("Mismatch in lengths: Unable to create DataFrame.")
    # Optionally exit the script
    exit()

df.dropna(inplace=True)

# Helper function to count the number of words
def word_count(sentence):
    return len(sentence.split())

df['num_words_article'] = df['article'].apply(word_count)
df['num_words_summary'] = df['summary'].apply(word_count)

# Removing outliers based on word counts (Optional)
df = df[(df['num_words_article'] <= 799) & (df['num_words_summary'] <= 350)]

# Sampling a small portion of the dataset for faster training
df = df.sample(frac=0.03)
df.reset_index(drop=True, inplace=True)

# Split dataset into train and test sets
train_size = 0.8
split_index = int(len(df) * train_size)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Convert dataframe to HuggingFace DatasetDict
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Load BART model and tokenizer
model_ckpt = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

# Preprocessing function for tokenizing the dataset
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["article"],
        max_length=799,
        truncation=True,
    )
    labels = tokenizer(
        examples["summary"], 
        max_length=350, 
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenizing dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(["path", "article", "summary", "num_words_article", "num_words_summary"])

# Create Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=True,  # Enable this if you're using GPU with mixed precision
)

# Load the ROUGE metric
rouge = evaluate.load('rouge')

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract the ROUGE scores
    return {k: round(v, 4) for k, v in result.items()}

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Evaluate the model on the test dataset
eval_results = trainer.evaluate()
print(eval_results)

# Summarization pipeline for generating summaries
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Example function to print an article, reference summary, and generated summary
def print_summary(idx):
    article = dataset["test"][idx]["article"]
    summary = dataset["test"][idx]["summary"]
    g_summary = summarizer(article)[0]["summary_text"]
    score = rouge.compute(predictions=[g_summary], references=[summary])
    print(f"Article: {article}")
    print(f"Reference Summary: {summary}")
    print(f"Generated Summary: {g_summary}")
    print(f"ROUGE Score: {score}")

# Generate summary and print ROUGE score for a test sample
print_summary(5)
