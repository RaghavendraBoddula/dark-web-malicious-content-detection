# Dark Web Malicious Content Detection

## Project Documentation and Current Code Implementation

This repository contains the project report supplied for the major project and the actual
Jupyter Notebook implementation supplied with it.

### What the supplied code currently implements

The supplied notebook implements a **TextCNN text-classification pipeline with pre-trained GloVe embeddings**.

Current pipeline:

1. Load `bbc-text.csv`
2. Clean text
3. Split into train/test sets
4. Label-encode categories
5. Tokenize and pad sequences
6. Load pre-trained GloVe 300-dimensional embeddings
7. Build a TextCNN using Conv1D layers and global max pooling
8. Train with Adam, early stopping and learning-rate reduction
9. Save the trained model, tokenizer and label encoder
10. Predict a class for new text

### Important implementation note

The project report describes a broader hybrid research approach using **TF-IDF + unsupervised LDA + GloVe + TextCNN**, including feature fusion of LDA document-topic vectors with CNN features.

The currently supplied notebook does **not** implement TF-IDF or LDA feature fusion. It implements the GloVe + TextCNN portion and currently points to a BBC text dataset.

This README intentionally documents the code that was actually supplied rather than claiming functionality that is not present in the code.

## Repository Structure

```text
dark-web-malicious-content-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── TextCNN_GloVe_Implementation.ipynb
├── src/
│   ├── textcnn_implementation.py
│   └── predict.py
├── data/
│   └── README.md
├── models/
│   └── README.md
├── database/
│   └── README.md
└── docs/
    └── Project_Report.pdf
```

## Setup

```bash
python -m venv venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Place the required dataset at:

```text
data/bbc-text.csv
```

and update the notebook's dataset path if necessary.

Download the required GloVe embedding file and place it at the path expected by the notebook:

```text
glove/glove.6B.300d.txt
```

## Run

Open:

```text
notebooks/TextCNN_GloVe_Implementation.ipynb
```

and run the cells in order.

The supplied notebook saves:

```text
textcnn_model.h5
textcnn_tokenizer.pkl
label_encoder.pkl
```

## Project Report

The complete supplied project report is available in:

`docs/Project_Report.pdf`

## Data and Security

- The original SQLite database is deliberately excluded because it contains Django authentication/session tables and credential-related records.
- Do not commit passwords, tokens, API keys, authentication databases or private datasets.
- Use only data that you are legally permitted to publish.

## Technologies

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, GloVe, TextCNN, NLP, Jupyter Notebook.
