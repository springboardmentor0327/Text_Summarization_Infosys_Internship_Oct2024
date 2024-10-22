# Text_Summarization_Infosys_Internship_Oct2024
This Text Summarization Project aims to develop a tool that efficiently condenses long articles, papers, or documents into concise summaries while preserving the key information and context. Utilizing advanced Natural Language Processing (NLP) techniques, this project focuses on both extractive and abstractive summarization methods.

# Original Summary
The global smartphone market experienced a significant shift in 2023, with total shipments reaching 1.21 billion units, a decline of 7.1% compared to the previous year. This was primarily due to supply chain disruptions, inflationary pressures, and geopolitical tensions. In contrast, Apple's market share grew from 23.4% in 2022 to 26.5% in 2023, driven by strong demand for the iPhone 14 series, particularly in North America and Europe. Meanwhile, Samsung's market share slightly decreased from 19.1% to 18.7%, despite launching the Galaxy S23 and Galaxy Z Fold 5. Xiaomi retained its third position globally, with an 11.8% market share, but also saw a 10% drop in shipments, largely due to a sluggish market in India. Additionally, 5G-enabled devices accounted for 70% of total smartphone shipments, up from 57% in 2022, reflecting increased consumer demand for faster mobile internet speeds. Analysts predict a rebound in 2024, with smartphone shipments expected to grow by 4.3%, as supply chain constraints ease and new innovations such as foldable screens and AI-powered features become more mainstream.

# T5 Generated Summary
The global smartphone market experienced a significant shift in 2023. Total shipments reached 1.21 billion units, a decline of 7.1% compared to the previous year. Analysts predict a rebound in 2024, with smartphone shipments expected to grow by 4.3%.

# BART Generated Summary
The global smartphone market experienced a significant shift in 2023, with total shipments reaching 1.21 billion units, a decline of 7.1% compared to the previous year. 5G-enabled devices accounted for 70% of total smartphone shipments, up from 57% in 2022,reflecting increased consumer demand for faster mobile internet speeds. Analysts predict a rebound in smartphone shipments in the coming year, with shipments expected to grow by 4.3%
    
# ROUGE Score table

| Model      | ROUGE-1           | ROUGE-2           | ROUGE-L           |
|:-----------|-------------------|:------------------|:------------------|
| BART       | 0.5433            | 0.4921            | 0.5118            |
| T5         | 0.3644            | 0.3318            | 0.3644            |

# Gradio

![Gradio](./images/BART-Text-summarisation.png)
