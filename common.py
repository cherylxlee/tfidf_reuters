from tfidf import *
from collections import Counter
import sys

"""
Print the most common 10 words from a documents and the word count.

1. Use gettext to get the text of the xml file.
2. Tokenize the text with tokenize.
3. Compute word counts with Counter.
4. Print most common words with counts.

Test results:
power 14
transmission 14
new 12
say 12
generator 12
electricity 11
cost 10
zealand 9
signal 8
charge 7
"""

path = sys.argv[1]

nlp = spacy.load("en_core_web_sm")

text = gettext(path)

tokens = tokenize(text, nlp)

counts = Counter(tokens)

most_common_words = counts.most_common(10)

for word, count in most_common_words:
    print(f"{word} {count}")

