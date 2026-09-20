# Dataset

The notebook currently expects a file named `bbc-text.csv` with at least these columns:

- `text`
- `category`

The dataset itself is not included in this repository. Place it in this folder (or update the path in the notebook) before running the training notebook.

Important: the current implementation is a TextCNN + GloVe text-classification pipeline using the BBC dataset path shown in the supplied notebook. It is not a Dark Web dataset.
