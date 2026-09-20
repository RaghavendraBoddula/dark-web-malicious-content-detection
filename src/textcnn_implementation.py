# ===== Notebook code cell 1 =====

# Required imports
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# ===== Notebook code cell 2 =====

# Load the dataset
file_path = "bbc-text.csv"  # Replace with the actual dataset path
dataset = pd.read_csv(file_path)

# Clean and preprocess text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

dataset['processed_text'] = dataset['text'].apply(preprocess_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    dataset['processed_text'], dataset['category'], test_size=0.2, random_state=42
)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Save the label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

# ===== Notebook code cell 3 =====

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform length
max_len = 200
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding="post")

# Save the tokenizer
joblib.dump(tokenizer, "textcnn_tokenizer.pkl")

# ===== Notebook code cell 4 =====

# Load pre-trained GloVe embeddings
def load_glove_embeddings(filepath, word_index, embedding_dim=300):
    embeddings_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    # Create embedding matrix
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Load GloVe embeddings (update the file path as necessary)
glove_path = "glove/glove.6B.300d.txt"  
embedding_dim = 300
embedding_matrix = load_glove_embeddings(glove_path, tokenizer.word_index, embedding_dim)

# ===== Notebook code cell 5 =====

# Define the TextCNN model
textcnn_model = Sequential([
    Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False  # Freeze embeddings
    ),
    Conv1D(filters=128, kernel_size=3, activation="relu"),
    Conv1D(filters=128, kernel_size=5, activation="relu"),
    GlobalMaxPooling1D(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation="softmax")
])

# Compile the model
textcnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5)

# Train the model
textcnn_model.fit(
    X_train_padded, y_train_encoded,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
textcnn_model.save("textcnn_model.h5")

# ===== Notebook code cell 6 =====

# Load the saved model and tokenizer
model = load_model("textcnn_model.h5")
tokenizer = joblib.load("textcnn_tokenizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define prediction function
def predict_text_with_class(text, model, tokenizer, label_encoder, max_len=200):
    # Preprocess the input text
    text = preprocess_text(text)
    # Tokenize and pad the text
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding="post")
    # Predict class
    predicted_index = model.predict(padded_seq).argmax(axis=1)[0]
    # Get class name
    predicted_class = label_encoder.inverse_transform([predicted_index])[0]
    return predicted_class

# Example: Predict class for a custom text
custom_text = ""
predicted_class = predict_text_with_class(custom_text, model, tokenizer, label_encoder)
print(f"Predicted Class: {predicted_class}")