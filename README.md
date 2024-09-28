# TFIDF with SpaCy for Document Summarization

![SpaCy](https://img.shields.io/badge/SpaCy-blue)
![TFIDF](https://img.shields.io/badge/TFIDF-brightgreen)
![lxml](https://img.shields.io/badge/lxml-yellow)

## Overview
This project applies **TFIDF (Term Frequency-Inverse Document Frequency)** to a collection of Reuters news articles to identify and rank the most important terms in each document. The project showcases text extraction, tokenization, and the computation of TFIDF scores using **SpaCy** for natural language processing. 

## Objective
The primary goal is to demonstrate how TFIDF can be used for text summarization by determining the most relevant terms in news articles. The project follows a bag-of-words model where word frequency and document frequency are analyzed across a large corpus of text.

## Key Features

### 1. Extracting and Parsing XML Data
The project begins by reading in XML files from a subset of Reuters articles. The text extraction is handled by using `lxml.etree` and XPath, allowing the script to efficiently retrieve the core text content of each article.

### 2. Tokenizing Text with SpaCy
The tokenization process is managed using **SpaCy**, with the following preprocessing steps:
- Converting text to lowercase.
- Removing punctuation, special characters, and numbers.
- Removing stopwords using SpaCy's built-in stopword list.
- Lemmatization to reduce words to their base forms.
- Filtering words with less than 3 characters.

### 3. Computing Term Frequencies and TFIDF Scores
Once the corpus is tokenized, the term frequencies are computed. TFIDF is then applied to score the relevance of each word, balancing its occurrence within a document against its frequency across the corpus. High TFIDF scores indicate words that are unique to a particular document.

### 4. Summarizing Articles
The summarization script uses TFIDF scores to output the top 20 terms for a given article, with words ranked by their relevance to the document. Words with scores below a threshold of 0.01 are discarded to focus on the most significant terms.

## Usage

### Tokenizing the Entire Corpus
Before summarizing, the entire corpus must be tokenized and stored. Use the following steps:

1. Download whatever dataset you are interested in (I used Reuters articles)
2. Run the script to tokenize the dataset and save the results to a pickle file:
   ```bash
   python compute_tok_corpus.py ~/data/reuters-vol1-disk1-subset
   ```

### Summarizing an Article
To summarize a specific article, load the tokenized corpus and compute the TFIDF scores:

```bash
python summarize.py ~/data/reuters-vol1-disk1-subset/33313newsML.xml
```
Example output for file `33313newsML.xml`:

```bash
generator 0.178
transmission 0.172
electricity 0.115
power 0.091
zealand 0.09
```

You can also compute the most common words in a document using the `common.py` script:

```bash
python common.py ~/data/reuters-vol1-disk1-subset/33313newsML.xml
```
Example Output:
```bash
power 14
transmission 14
new 12
say 12
generator 12
```

## Testing
You can run tests to verify the accuracy of the implemented TFIDF functions using pytest:

```bash
pytest -vv test_tfidf.py
```

## Components
- `tfidf.py`: Contains methods like gettext(), tokenize(), doc_freq(), and summarize() for processing the text.
- `common.py`: A script that prints the 10 most common words in an article.
- `compute_tok_corpus.py`: Tokenizes the entire corpus and saves it to a pickle file.
- `summarize.py`: Outputs the top 20 terms by TFIDF score for a specific document.

## Libraries Used
- `SpaCy` for tokenization and lemmatization.
- `lxml` for XML parsing.
- `Counter` for word frequency analysis.
- Python 3.8+

## Notes
The dataset used for this project is a subset of Reuters news articles, which are processed in XML format. Due to licensing restrictions, this dataset cannot be shared publicly.
The TFIDF implementation allows for customizable thresholds and can be adapted for different types of corpora.
