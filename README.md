# 🎙️ Speaker Diarization — ANN Project

> **"Who Spoke When?"** — An end-to-end speaker diarization pipeline on the AMI Meeting Corpus using CNN triplet embeddings, a Transformer encoder, and WavLM + Whisper combined representations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Project Structure](#project-structure)
- [Models](#models)
  - [CNN Encoder (Triplet Loss)](#1-cnn-encoder-triplet-loss)
  - [Transformer Encoder](#2-transformer-encoder)
  - [WavLM + Whisper](#3-wavlm--whisper-combined)
- [Results](#results)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Usage](#usage)
- [Future Work](#future-work)
- [Team](#team)

---

## Overview

Speaker diarization is the task of partitioning an audio stream into homogeneous segments according to speaker identity. This project builds a full diarization pipeline from scratch on multi-party meeting recordings, experimenting with three different embedding strategies and evaluating using Diarization Error Rate (DER).

**Key contributions:**
- Sliding-window audio segmentation with silence detection (RMS thresholding)
- CNN encoder trained with triplet loss to produce 256-d speaker embeddings
- Transformer encoder operating on sequences of CNN embeddings for temporal context
- WavLM + Whisper combined 1280-d representations
- Agglomerative clustering for speaker assignment
- Whisper ASR for per-speaker transcript generation
- Full evaluation against AMI ground-truth XML annotations

---

## Dataset

**AMI Meeting Corpus** — 10 multi-party meeting recordings

| Property | Value |
|---|---|
| Meetings used | 10 |
| Speakers per meeting | 4–5 |
| Audio sample rate | 16 kHz |
| Avg. meeting length | ~30 minutes |
| Annotation format | XML (`.segments` files) |
| Split | 7 train / 1 val / 2 test |

**Meetings:**

| Meeting | Split |
|---|---|
| EN2001a, EN2002a, EN2003a, EN2004a, EN2005a, EN2009b, IB4001 | TRAIN |
| IN1001 | VAL |
| IS1000a, TS3003a | TEST |

Download the AMI corpus from [https://groups.inf.ed.ac.uk/ami/corpus/](https://groups.inf.ed.ac.uk/ami/corpus/) and place it under `data/AMI_Corpus/`.

---

## Pipeline

```
Audio Input (WAV, 16 kHz)
        │
        ▼
Preprocessing (mix channels, normalize)
        │
        ▼
Sliding Window Segmentation
  • Window: 30 seconds
  • Hop:    15 seconds
  • Overlap: 50%
        │
        ▼
Voice Activity Detection
  • RMS energy threshold: 0.00061
  • Remove silent windows
        │
        ├──────────────────────┐
        ▼                      ▼
XML Label Parsing         Feature Extraction
(ground truth)            Log-Mel (128 bands)
                          WavLM (768-d)
                          Whisper (512-d)
        │
        ▼
Embedding Model
  ┌──────────────┐
  │ CNN Encoder  │  → 256-d per window
  │ Transformer  │  → 256-d with context
  │ WavLM+Whisper│  → 1280-d combined
  └──────────────┘
        │
        ▼
Agglomerative Clustering (Ward linkage)
  • Cosine similarity space
  • k = number of speakers
        │
        ▼
Timeline Reconstruction
  • Assign speaker labels to time segments
        │
        ▼
Whisper ASR
  • Transcribe each labeled segment
  • Word-level alignment to speakers
        │
        ▼
Evaluation (DER)
  + Final Output CSV
```

---

## Project Structure

```
Speaker_Diarization/
│
├── data/
│   ├── AMI_Corpus/                  # Raw meeting audio + XML annotations
│   └── ami_public_manual_1.6.2/     # Manual annotation files
│
├── notebooks/
│   ├── s1.ipynb                     # Preprocessing & feature extraction (HuBERT path)
│   ├── s4.ipynb                     # Preprocessing variant (WavLM path)
│   ├── s1_with_whisper.ipynb        # Preprocessing + Whisper feature extraction
│   ├── CNN_Embeddings_Speaker_      # CNN encoder training + clustering
│   │   diarization.ipynb
│   ├── Transformer_Whisper.ipynb    # Whisper ASR + speaker alignment
│   └── s1_final_trained_            # Final WavLM + Whisper + Transformer pipeline
│       transformer_WavLM_Whisper.ipynb
│
├── outputs/
│   ├── diarization_output.csv       # Diarized segments (meeting, start, end, speaker)
│   ├── all_transcripts.csv          # Full transcripts with speaker labels
│   ├── final_results.csv            # DER scores per meeting per model
│   └── pipeline_diagram.png         # Pipeline visualization
│
├── models/
│   ├── transformer_best.pth         # Best Transformer encoder checkpoint
│   └── diarization_pipeline.pth     # Full pipeline checkpoint
│
└── README.md
```

---

## Models

### 1. CNN Encoder (Triplet Loss)

A convolutional network trained to map 30-second audio windows (represented as 128-band log-mel spectrograms) to a 256-dimensional embedding space where same-speaker segments cluster together.

**Architecture:**

```
Input: Log-Mel Spectrogram (1 × 128 × T)
  └─ Conv2d(1→32, 3×3) + BatchNorm + ReLU + MaxPool(2)
  └─ Conv2d(32→64, 3×3) + BatchNorm + ReLU + MaxPool(2)
  └─ Conv2d(64→128, 3×3) + BatchNorm + ReLU + MaxPool(2)
  └─ AdaptiveAvgPool → Flatten
  └─ Linear(128×16×16 → 256) → L2 Normalize
Output: 256-d speaker embedding
```

**Training:**

| Hyperparameter | Value |
|---|---|
| Loss | Triplet Margin Loss (margin = 0.3) |
| Optimizer | Adam (lr = 1e-4) |
| Batch size | 64 |
| Epochs | 20 (early stopping, patience = 4) |
| Embedding dim | 256 |

**Why triplet loss?** It pulls embeddings from the same speaker together and pushes embeddings from different speakers apart, without needing hard class labels — just anchor/positive/negative triplets.

---

### 2. Transformer Encoder

A Transformer encoder that refines CNN embeddings by attending over a 30-second context window (10 consecutive 3-second CNN chunks). This allows the model to leverage temporal context — a speaker's identity across a sequence of windows — rather than treating each window independently.

**Architecture:**

```
Input: Sequence of CNN embeddings (B × 10 × 256)
  └─ Linear projection (256 → 256)
  └─ Positional Encoding (Embedding, max_len=512)
  └─ TransformerEncoderLayer × 2
       nhead=4, d_feedforward=512, dropout=0.3
  └─ Per-token output: 256-d
```

**Training:**

| Hyperparameter | Value |
|---|---|
| Loss | Sequence Triplet Loss (in-batch, margin = 0.3) |
| Optimizer | Adam (lr = 1e-4) + ReduceLROnPlateau |
| Epochs | 30 (early stopping, patience = 8) |
| Context | 30 seconds (10 × 3s CNN chunks) |
| Best val loss | 0.1904 |

---

### 3. WavLM + Whisper (Combined)

Instead of using CNN-extracted features, this approach extracts embeddings directly from two pre-trained models and concatenates them for a richer 1280-dimensional representation.

| Model | Source | Output | Purpose |
|---|---|---|---|
| WavLM | `microsoft/wavlm-base-plus` | 768-d | Speaker identity, acoustic features |
| Whisper | `openai/whisper-base` (encoder) | 512-d | Linguistic + acoustic context |
| **Combined** | Concatenation | **1280-d** | Rich speaker representation |

**Why WavLM?** Pre-trained on 94k hours of speech with an objective specifically designed to model speaker identity and overlapping speech — state of the art for speaker verification tasks.

**Why Whisper?** Its encoder captures both acoustic and linguistic context, giving the model information about *what* is being said as well as *how* it sounds — complementary to WavLM's speaker-identity focus.

---

## Results

Evaluation metric: **Diarization Error Rate (DER)**

```
DER = (False Alarm + Missed Speech + Speaker Confusion) / Total Speech Duration
```
Lower is better.

| Meeting | Split | DER CNN | DER Transformer | DER Smoothed |
|---|---|---|---|---|
| EN2001a | TRAIN | 26.5% | 55.8% | 57.1% |
| EN2002a | TRAIN | 47.0% | 53.7% | 48.6% |
| EN2003a | TRAIN | 30.4% | 35.4% | **28.2%** |
| EN2004a | TRAIN | **35.7%** | 43.8% | 55.0% |
| EN2005a | TRAIN | **48.9%** | 59.6% | 42.9% |
| EN2009b | TRAIN | **39.2%** | 51.5% | 35.0% |
| IB4001 | TRAIN | 46.8% | 46.0% | **40.6%** |
| IN1001 | VAL | **51.6%** | 56.7% | 50.0% |
| IS1000a | TEST | **51.2%** | 71.0% | 60.0% |
| TS3003a | TEST | **37.5%** | 63.4% | 52.0% |

**Average DER:**

| Model | Avg DER |
|---|---|
|  CNN (best) | **41.5%** |
| Smoothed | 47.9% |
| Transformer | 54.5% |

The CNN encoder with triplet loss achieved the best average DER across all meetings, suggesting that for this dataset size, simpler embeddings with direct clustering outperform the Transformer's added complexity.

---

## Outputs

### `diarization_output.csv`
Speaker timeline for each meeting.

```
meeting, start, end, speaker, duration, split
EN2001a, 1.5,   9.0,  Speaker_4, 7.5, TRAIN
EN2001a, 9.0,   18.0, Speaker_2, 9.0, TRAIN
...
```

### `all_transcripts.csv`
Word-aligned speaker transcripts generated by Whisper.

```
meeting, start,  end,   speaker,   text
EN2001a, 2.98,   5.94,  Speaker_4, "Okay. Okay."
EN2001a, 11.12,  17.72, Speaker_2, "Does anyone want to see Steve's feedback..."
...
```

---

## Requirements

```
torch>=2.0
torchaudio
transformers
openai-whisper
librosa
soundfile
numpy
pandas
scikit-learn
tqdm
matplotlib
umap-learn
```

Install with:
```bash
pip install torch torchaudio transformers openai-whisper librosa soundfile numpy pandas scikit-learn tqdm matplotlib umap-learn
```

---

## Usage

### 1. Preprocessing

Run `s1.ipynb` (or `s4.ipynb` for the WavLM path) to:
- Load AMI meetings
- Apply sliding-window segmentation (30s / 15s hop)
- Filter silent windows via RMS threshold
- Save audio chunks and log-mel features

### 2. CNN Training + Diarization

Run `CNN_Embeddings_Speaker_diarization.ipynb` to:
- Parse XML speaker annotations
- Build the labeled CNN dataset
- Train the `CNNEncoder` with triplet loss
- Extract embeddings and run agglomerative clustering
- Compute DER vs. ground truth

### 3. Transformer Training

Run after CNN: the Transformer takes sequences of CNN embeddings as input.
See the Transformer notebook for training and UMAP visualization of embeddings.

### 4. WavLM + Whisper Pipeline

Run `s1_final_trained_transformer_WavLM_Whisper.ipynb` to:
- Extract WavLM (768-d) and Whisper encoder (512-d) features
- Concatenate → 1280-d vectors
- Train the Transformer classifier on combined embeddings
- Evaluate DER

### 5. Transcript Generation

Run `Transformer_Whisper.ipynb` to:
- Transcribe each meeting with Whisper (`base` model)
- Align word-level timestamps to diarization output
- Group words into utterances per speaker
- Export `all_transcripts.csv`

---

## Future Work

1. **End-to-end learning** — Train a unified model for segmentation, embedding, and clustering jointly to reduce error accumulation across stages.
2. **Overlapping speech detection** — Add a dedicated head for detecting and handling simultaneous speakers.
3. **Larger datasets** — Evaluate on the full AMI corpus (170+ meetings), CALLHOME, VoxConverse, and CHiME-6.
4. **Domain adaptation** — Fine-tune on telephone calls, medical interviews, or broadcast speech.
5. **Automatic speaker count estimation** — Replace fixed `k` with BIC, eigen-gap, or a learned threshold for fully unsupervised diarization.
6. **Real-time inference** — Optimize for streaming/online diarization with chunk-based processing.

---

## Team
1-Maya Anwar
2-Nada Ibrahim
3-Steven Willson
4-Abdallah khaled
5-Youssef Ahmed 

**ANN Project — Speaker Diarization**
GitHub: [github.com/Nada-Elghaweet/Speaker_Diarization](https://github.com/Nada-Elghaweet/Speaker_Diarization)
