import csv
import tensorflow as tf
import nltk
nltk.download('punkt')
from collections import Counter
from nltk.tokenize import word_tokenize

# Read product data from CSV file
products = []
with open('products.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        products.append(row)

# Create a tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Process text data for each product
for product in products:
    print(product)
    # Preprocess product name (lowercase, remove punctuation, etc.)
    name = product['název'].lower()  # Example preprocessing, adjust as needed
    
    # Update tokenizer with product name
    tokenizer.fit_on_texts([name])

# Get the word frequencies from the tokenizer
word_freq = tokenizer.word_counts

# Print the word frequencies
for word, freq in word_freq.items():
    print(f'Word: {word}, Frequency: {freq}')

# Tokenization and vocabulary creation simply from product description
for product in products:
    tokenized_text = [nltk.word_tokenize(sentence.lower()) for sentence in product['popis']]
    word_counts = Counter(word for sentence in tokenized_text for word in sentence)
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}

# Print vocabulary
print("Vocabulary:")
print(vocab)

# Initialize a list to store tokens
tokens = []

# Tokenize each sentence and collect tokens
for product in products:
    tokens.extend(word_tokenize(product['popis']))

# Remove duplicate tokens to create the vocabulary
vocabulary = list(set(tokens))

# Print the vocabulary
print(vocabulary)