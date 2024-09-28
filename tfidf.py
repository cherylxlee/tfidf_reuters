import sys

import spacy
from lxml import etree
from collections import Counter
import string
import zipfile
import os
import numpy as np
import re

"""
pytest -v

test_tfidf.py::test_gettext PASSED                                                              [ 14%]
test_tfidf.py::test_tokenize PASSED                                                             [ 28%]
test_tfidf.py::test_tokenize_2 PASSED                                                           [ 42%]
test_tfidf.py::test_doc_freq PASSED                                                             [ 57%]
test_tfidf.py::test_compute_tfidf_i PASSED                                                      [ 71%]
test_tfidf.py::test_compute_tfidf PASSED                                                        [ 85%]
test_tfidf.py::test_summarize PASSED                                                            [100%]
"""

def gettext(xmlfile) -> str:
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    root = etree.parse(xmlfile)
    title = root.xpath('//title/text()')
    text = root.xpath('//text/p/text()')
    
    title_str = ' '.join(title) if title else ''
    text_str = ' '.join(text) if text else ''
    return (title_str + ' ' + text_str).strip()


def tokenize(text, nlp) -> list:
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. 
      1. Normalize to lowercase. Strip punctuation, numbers, and special characters 'r`, `n`, `t`. 
      2. Replace multiple spaces for a single space.
      3. Tokenize with spacy.
      4. Remove stopwords with spacy.
      5. Remove tokens with len <= 2.
      6. Apply lemmatization to words using spacy.
    """
    text = text.lower()
    text = re.sub(r'[' + re.escape(string.punctuation) + r'0-9\r\t\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    doc = nlp(text)
    
    tokens = [token.lemma_ for token in doc if not token.is_stop 
              and len(token.lemma_) > 2 and token.is_alpha]
    
    return tokens


def doc_freq(tok_corpus):
    """
    Returns a dictionary of the number of docs in which a word occurs.
    Input:
       tok_corpus: list of list of words
    Output:
       df: dictionary df[w] = # of docs containing w 
    """
    df = Counter()
    
    for doc in tok_corpus:
        unique_tokens = set(doc)
        df.update(unique_tokens)
    
    return dict(df)


def compute_tfidf_i(tok_doc: list, doc_freq: dict, N: int) -> dict:
    """ Returns a dictionary of tfidf for one document
        tf[w, doc] = counts[w, doc]/ len(doc)
        idf[w] = np.log(N/(doc_freq[w] + 1))
        tfidf[w, doc] = tf[w, doc]*idf[w]
    """
    tfidf = {}
    doc_length = len(tok_doc)
    counts = Counter(tok_doc)
    
    for word, count in counts.items():
        tf = count / doc_length 
        idf = np.log(N / (doc_freq.get(word, 0) + 1))
        tfidf[word] = tf * idf
    
    return tfidf


def compute_tfidf(tok_corpus:list, doc_freq: dict) -> dict:
    """Computes tfidf for a corpus of tokenized text.

    Input:
       tok_corpus: list of tokenized text
       doc_freq: dictionary of word to set of doc indeces
    Output:
       tfidf: list of dict 
               tfidf[i] is the dictionary of tfidf of word in doc i.
    """
    N = len(tok_corpus)
    tfidf_corpus = []
    
    for tok_doc in tok_corpus:
        tfidf = compute_tfidf_i(tok_doc, doc_freq, N)
        tfidf_corpus.append(tfidf)
    
    return tfidf_corpus


def summarize(xmlfile, doc_freq, N,  n:int) -> list:
    """
    Given xml file, n and the tfidf dictionary 
    return up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.01. Sort the (word,score) pairs by TFIDF score in reverse order.
    if words have the same score, they should be sorted in alphabet order.
    """
    text = gettext(xmlfile)
    nlp = spacy.load("en_core_web_sm")
    tokens = tokenize(text, nlp)
    
    tfidf = compute_tfidf_i(tokens, doc_freq, N)
    
    filtered_tfidf = {word: score for word, score in tfidf.items() if score >= 0.01}
    sorted_tfidf = sorted(filtered_tfidf.items(), key=lambda x: (-x[1], x[0]))

    return sorted_tfidf[:n]

