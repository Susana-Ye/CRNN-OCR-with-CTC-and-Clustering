# CRNN-Based OCR with CTC Loss and Logit Clustering

This project implements an end-to-end **scene text recognition (OCR)** pipeline using a **Convolutional Recurrent Neural Network (CRNN)** trained with **Connectionist Temporal Classification (CTC) loss**.  
In addition to text recognition, the project includes a post-hoc **clustering analysis of model logits** to study uncertainty and confusion patterns in the learned representations.

> Originally developed as part of a university coursework and later refined for portfolio presentation.


## Overview

The pipeline consists of:
- A **CNN backbone** for visual feature extraction from variable-width text images
- **Bidirectional LSTM layers** to model sequential dependencies along the image width
- **CTC loss** to handle alignment between image features and text labels without character-level annotations
- **CTC decoding** (greedy and beam search)
- **Clustering analysis** on per-time-step logits using KMeans, including Silhouette Score and Purity metrics


## Model Architecture

- **CNN**: Deep convolutional backbone inspired by standard CRNN architectures for OCR
- **Sequence Modeling**: 2-layer Bidirectional LSTM
- **Loss**: Connectionist Temporal Classification (CTC)
- **Decoding**: Greedy decoding and beam search
- **Clustering**: KMeans on standardized logit vectors (K = number of character classes)


## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, SciPy
- scikit-learn
- Pillow
- tqdm


## Project Structure

```
.
├─ notebooks/
│  └─ pa2.ipynb
├─ results/              # figures, error examples, metrics (optional)
├─ data/                 # dataset directory (not included)
├─ environment.yml
├─ requirements.txt      # optional
└─ README.md
```


## Setup (Conda – recommended)

```bash
conda env create -f environment.yml
conda activate crnn-ocr
jupyter notebook
```


## Setup (pip – optional)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```


## Usage

1. Place the dataset inside the `data/` directory.
2. Open `notebooks/pa2.ipynb`.
3. Run all cells to train, evaluate, and perform clustering analysis.



## Dataset Notes

- The dataset is **not included** in this repository.
- Images consist of scene text with variable widths.
- The pipeline supports variable-length sequences through padded batching and CTC-compatible formatting.


## Evaluation

Model performance is evaluated using:
- Exact-match **accuracy** on the test set
- Qualitative inspection of error cases
- **Silhouette Score** on clustered logits
- **Cluster Purity** with respect to greedy CTC predictions



## Key Takeaways

- CTC enables sequence-to-sequence learning without explicit alignment.
- Bidirectional RNNs effectively capture character-level context in OCR tasks.
- Clustering logits reveals structured uncertainty and common confusion modes beyond simple argmax decoding.


## Academic Context

This project was completed as part of a Machine Learning course assignment and represents independent work.  
The repository contains only original code and does not include assignment prompts, solutions, or proprietary datasets.

