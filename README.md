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

