# Experimental English ASR

An experimental English Automatic Speech Recognition (ASR) pipeline built from scratch using **PyTorch**, **CNN**, **BiLSTM**, and **Connectionist Temporal Classification (CTC)** on the **LibriSpeech** dataset.

The main goal of this project was to understand and implement the complete ASR pipeline, from raw speech preprocessing to model training, decoding, evaluation, and error analysis.

> **Project Type:** Experimental / Educational
> **Language:** English
> **Framework:** PyTorch
> **Dataset:** LibriSpeech
> **Task:** Speech-to-Text

---

## Overview

This project implements a complete CTC-based English ASR pipeline without relying on a pretrained speech recognition model.

The system covers:

* LibriSpeech dataset preparation
* Audio preprocessing
* Mel-Spectrogram feature extraction
* Custom PyTorch Dataset and DataLoader
* Variable-length batch processing
* CNN-based acoustic feature extraction
* Bidirectional LSTM sequence modeling
* CTC-based sequence alignment
* Greedy CTC decoding
* Checkpoint saving and recovery
* CER / WER evaluation
* Per-sample prediction analysis
* Error analysis
* Unit testing

The project is intentionally experimental and was developed to build a deeper understanding of the internal components of an ASR system.

---

## Architecture

```text
                    Raw Audio
                        │
                        ▼
              Audio Preprocessing
                        │
                        ▼
                Mel-Spectrogram
                        │
                        ▼
                   CNN Encoder
                        │
                        ▼
                  BiLSTM Encoder
                        │
                        ▼
                   Linear Layer
                        │
                        ▼
                  CTC Logits
                        │
                        ▼
              Greedy CTC Decoder
                        │
                        ▼
                   Transcript
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
             CER                 WER
              │                   │
              └─────────┬─────────┘
                        ▼
                  Error Analysis
```

---

## Model

The acoustic model consists of three main components.

### CNN Feature Extractor

Two convolutional blocks are used to extract local acoustic patterns from the Mel-Spectrogram.

```text
Conv2D
BatchNorm
ReLU
MaxPool
```

The frequency dimension is progressively reduced while preserving the temporal dimension.

### BiLSTM Sequence Model

The CNN output is converted into a temporal sequence and processed using a two-layer bidirectional LSTM.

The bidirectional structure allows the model to use contextual information from both past and future frames during training.

### CTC Classifier

The BiLSTM output is passed through a linear classification layer.

The model predicts character-level probabilities over:

* Blank token
* Space
* Apostrophe
* A-Z

A CTC loss function is then used to align the input acoustic sequence with the target transcript without requiring frame-level alignment.

---

## Audio Preprocessing

Audio is converted into a consistent representation before entering the model.

### Configuration

```text
Sample Rate: 16000 Hz
FFT Size: 400
Hop Length: 160
Mel Bins: 80
Power: 2.0
Feature: Log-Mel Spectrogram
```

Variable-length audio samples are padded during batch creation.

The preprocessing pipeline also handles:

* Tensor conversion
* Channel handling
* Resampling
* Mel-Spectrogram extraction
* Amplitude-to-dB conversion

---

## Dataset

The project uses **LibriSpeech** for English speech recognition.

The dataset pipeline includes:

```text
Raw Audio
    ↓
Transcript Files
    ↓
Manifest Generation
    ↓
CSV Metadata
    ↓
PyTorch Dataset
    ↓
DataLoader
```

Raw dataset files are intentionally excluded from the repository.

---

## Training

The model is trained using:

```text
Optimizer: Adam
Loss: CTC Loss
Training: GPU / CUDA when available
```

Checkpoint recovery is supported so that training can resume from the latest saved state.

Two types of checkpoints are maintained:

* Best model checkpoint
* Resumable training checkpoint


---

## Evaluation

The trained model is evaluated on the LibriSpeech test set using greedy CTC decoding.

### Evaluation Metrics

Two standard ASR metrics are used:

**Character Error Rate (CER)**

Measures character-level transcription errors.

**Word Error Rate (WER)**

Measures word-level transcription errors.

### Test Results

```text
Test Samples: 2620

Corpus CER: 14.69%
Corpus WER: 44.31%
```

The evaluation pipeline also stores per-sample:

```text
Reference
Prediction
CER
WER
Utterance ID
Speaker ID
```

in:

```text
results/evaluation/test_predictions.csv
```

---

## Error Analysis

The project includes an error analysis pipeline based on word-level alignment.

The evaluation analyzes:

* Substitutions
* Deletions
* Insertions
* Exact matches
* Per-sample CER
* Per-sample WER
* High-error predictions

For the evaluated test set, the dominant error category was **word substitution**, indicating that the model frequently produced incorrect words rather than simply omitting or inserting words.

This analysis is intended to help identify weaknesses in the experimental architecture.

---

## Project Structure

```text
experimental-english-asr/
│
├── data/
│   ├── raw/
│   └── processed/
│       └── manifests/
│
├── results/
│   └── evaluation/
│       └── test_predictions.csv
│
├── src/
│   ├── data/
│   │   └── Librispeech_dataset.py
│   │
│   ├── preprocessing/
│   │   └── audio_preprocessing.py
│   │
│   ├── models/
│   │   └── ...
│   │
│   ├── training/
│   │   ├── ctc_train.py
│   │   └── benchmark_dataloader.py
│   │
│   ├── evaluation/
│   │   ├── evaluation_asr.py
│   │   ├── metrics_asr.py
│   │   └── error_analysis.py
│   │
│   └── tests/
│       ├── test_greedy_decoder.py
│       └── test_metrics_asr.py
│
├── .gitignore
└── README.md
```

---

## Technologies

* Python
* PyTorch
* Torchaudio
* Librosa / Audio processing tools
* NumPy
* Pandas
* JiWER
* CUDA
* Git / GitHub

---

## Limitations

This project
