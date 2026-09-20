# Dark Web Malicious Content Detection

A hybrid NLP and deep-learning framework for classifying potentially illicit Dark Web text.

## Project Overview

The project addresses the need for efficient and accurate text classification for monitoring and threat-intelligence use cases involving Dark Web content.

The proposed framework combines:

- TF-IDF
- Unsupervised Latent Dirichlet Allocation (LDA)
- Pre-trained GloVe embeddings
- TextCNN
- LDA document-topic vectors fused with CNN feature representations

The goal is to combine global thematic information with local textual features to improve semantic understanding, robustness, and classification performance.

## Problem Statement

Existing approaches that combine TextCNN with Labeled Latent Dirichlet Allocation (LLDA) depend heavily on class-dependent topic modelling and keyword-based embedding construction. The project document identifies limitations including restricted vocabulary coverage, loss of semantic information, and limited adaptability to emerging Dark Web terminology.

## Proposed Approach

The proposed pipeline is:

```text
Dark Web Text
     |
     v
Text Preprocessing
     |
     +----> TF-IDF --------------------+
     |                                  |
     +----> Unsupervised LDA -----------+----> Feature Fusion
     |                                  |
     +----> GloVe Embeddings -----------+
                                        |
                                        v
                                     TextCNN
                                        |
                                        v
                                  Classification
```

### 1. TF-IDF

TF-IDF is used to emphasize informative terms in the text.

### 2. Unsupervised LDA

LDA is used to capture latent thematic structures without depending on class labels.

### 3. GloVe Embeddings

Pre-trained GloVe embeddings provide semantic representations of words.

### 4. TextCNN

TextCNN extracts discriminative local features from the weighted embedding sequences.

### 5. Feature Fusion

The LDA document-topic vectors are fused with CNN feature representations to incorporate both global and local contextual information.

## Research Outcome

According to the supplied project abstract, experimental analysis indicates that the multi-feature fusion approach improves robustness and semantic understanding and achieves higher classification accuracy than the existing LLDA-guided TextCNN model.

No numerical accuracy, dataset size, benchmark score, or detailed experimental configuration is stated in the supplied project document, so this repository does not invent those values.

## Key Technologies / Concepts

- Natural Language Processing
- Text Classification
- Deep Learning
- TextCNN
- TF-IDF
- Latent Dirichlet Allocation (LDA)
- GloVe Word Embeddings
- Topic Modeling
- Illicit Content Detection
- Dark Web Threat Intelligence

## Project Structure

```text
dark-web-malicious-content-detection/
|
├── README.md
├── docs/
│   └── Project_Report.pdf
└── .gitignore
```

## Documentation

The `docs/Project_Report.pdf` file contains the project material supplied for this repository, including:

- Abstract
- Project introduction and objectives
- Literature survey
- Existing and proposed system
- Functional and non-functional requirements
- Feasibility analysis
- System architecture
- Data flow diagram
- UML diagrams
- Conclusion

## Important Note

The supplied project material describes the methodology and project documentation, but it does not include the complete source-code implementation or dataset. Those should only be added to this repository when the original project source files are available and are appropriate for public sharing.

## Keywords

Dark Web Classification, TF-IDF, LDA, GloVe Embeddings, TextCNN, Topic Modeling, Deep Learning, Illicit Content Detection
