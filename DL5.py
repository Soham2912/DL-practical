import numpy as np
import re
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from sklearn.decomposition import PCA
import seaborn as sns

# Sample data
data = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance."""

# Split into sentences
sentences = data.split('.')
clean_sent = []

for sentence in sentences:
    if sentence == "":
        continue
    sentence = re.sub('[^A-Za-z0-9]+', ' ', sentence)
    sentence = re.sub(r'(?:^| )\w(?:$| )', ' ', sentence).strip()
    sentence = sentence.lower()
    clean_sent.append(sentence)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_sent)
sequences = tokenizer.texts_to_sequences(clean_sent)

# Create word-to-index and index-to-word mappings
index_to_word = {v: k for k, v in tokenizer.word_index.items()}
word_to_index = tokenizer.word_index

# Define vocabulary size, embedding size, and context size
vocab_size = len(tokenizer.word_index) + 1
emb_size = 10
context_size = 2
contexts = []
targets = []

# Prepare context-target pairs
for sequence in sequences:
    for i in range(context_size, len(sequence) - context_size):
        target = sequence[i]
        context = [sequence[i - 2], sequence[i - 1], sequence[i + 1], sequence[i + 2]]
        contexts.append(context)
        targets.append(target)

# Display example context-target pairs
for i in range(min(5, len(contexts))):
    words = [index_to_word.get(idx, "") for idx in contexts[i]]
    target_word = index_to_word.get(targets[i], "")
    print(words, " -> ", target_word)

# Convert contexts and targets to arrays for model input
X = np.array(contexts)
Y = np.array(targets)

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=emb_size, input_length=2 * context_size),
    Lambda(lambda x: tf.reduce_mean(x, axis=1)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Compile and train the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=80)

# Plot training accuracy
sns.lineplot(data=history.history['accuracy'])

# Embedding visualization
embeddings = model.layers[0].get_weights()[0]
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Testing predictions on test sentences
test_sentences = [
    "known as structured learning",
    "transformers have applied to",
    "where they produced results",
    "cases surpassing expert performance"
]

for sent in test_sentences:
    test_words = sent.split()
    x_test = [word_to_index.get(word) for word in test_words if word_to_index.get(word) is not None]
    if len(x_test) == 2 * context_size:
        x_test = np.array([x_test])
        pred = model.predict(x_test)
        pred_index = np.argmax(pred[0])
        print(f"Prediction for '{sent}': {index_to_word.get(pred_index, 'unknown word')}")
    else:
        print(f"Insufficient context for sentence '{sent}'")
