import tensorflow as tf
import psycopg2
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# preprocess and encode product data into embeddings using TensorFlow

# Define parameters for embedding layer
vocab_size = 10000  # Example vocabulary size
embedding_dim = 100  # Example embedding dimension
max_sequence_length = 100  # Example maximum sequence length

# Create tokenizer
with open('products.csv', newline='') as csvfile:
    products = csv.reader(csvfile, delimiter=',')
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    texts = [product[2] for product in products]
    tokenizer.fit_on_texts(texts)

# define a function to preprocess text data and convert it into embeddings
def preprocess_and_encode(text):
     # Convert text to sequence of indices
    sequences = tokenizer.texts_to_sequences([text])

    # Pad sequences to ensure uniform length
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    # Encode text into embeddings using TensorFlow
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(padded_sequences)
    return embedding

# Function to convert embedding tensor to binary blob
def tensor_to_blob(embedding_tensor):
    return psycopg2.Binary(embedding_tensor.numpy().tobytes())

# connect to PostgreSQL database
conn = psycopg2.connect("host=db dbname=database_dev user=postgres password=secret")

# insert product data and embeddings into PostgreSQL database
cursor = conn.cursor()
with open('products.csv', newline='') as csvfile:
    products = csv.reader(csvfile, delimiter=',')

    # Skip the first row
    next(products)


    for product in products:
        embedding = preprocess_and_encode(product[2])
        embedding_blob = tensor_to_blob(embedding)
        cursor.execute("INSERT INTO products (id, name, description, category, embedding) VALUES (%s, %s, %s, %s, %s)",
                    (product[0], product[1], product[2], product[3], embedding_blob))
conn.commit()
cursor.close()

# create fulltex index
cursorIndex = conn.cursor()
cursorIndex.execute("CREATE INDEX products_idx ON products USING GIN (to_tsvector('english', name || ' ' || description || ' ' || category))")
conn.commit()
cursorIndex.close()
conn.close()
