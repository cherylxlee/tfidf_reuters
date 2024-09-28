from tfidf import *
import os
import pickle

"""
1. Get a list of all xml_files in the corpus.
2. Get a list of texts for all files xml_files.
3. Get a list of tokenized text (list of list of tokens).
4. Save the tokenized corpus in ~.data.tok_corpus.pickle.
"""
directory_path = sys.argv[1]

nlp = spacy.load("en_core_web_sm")

xml_files = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".xml")]

texts = [gettext(xml_file) for xml_file in xml_files]

tok_corpus = [tokenize(text, nlp) for text in texts]

pickle_file = os.path.expanduser("~/data/tok_corpus.pickle")
with open(pickle_file, 'wb') as file:
    pickle.dump(tok_corpus, file)
