# Speaker Diarization on the AMI Corpus

A speaker diarization and transcription pipeline built on the AMI Meeting Corpus. Four independent model experiments share a common preprocessing and segmentation backbone, and each produces a speaker-attributed transcript by pairing a diarization model with Whisper.

---

## Team

| Member | Approach |
|---|---|
| Maya, Steven | HuBERT + Transformer + Whisper |
| Youssef | WavLM + Whisper |
| Abdellah | HuBERT + Whisper |
| Nada | CNN (from scratch) + Transformer (from scratch) + Whisper |

---

## Dataset

**AMI Meeting Corpus** — multi-party meeting recordings with per-speaker headset audio and manual segment annotations.

- Nada's experiments: 10 meetings (`EN2001a`, `EN2002a`, `EN2003a`, `EN2004a`, `EN2005a`, `EN2009b`, `IB4001`, `IN1001`, `IS1000a`, `TS3003a`)
- Maya/Steven's experiments: 30 meetings (EN and ES series)
- Annotations: `ami_public_manual_1.6.2` segment XML files

---

## Shared Pipeline

All experiments follow the same preprocessing and segmentation steps before branching into their respective models.

```
Raw WAV (per-speaker headset)
        |
        v
Mix headset channels -> normalize
        |
        v
Sliding window segmentation
        |
        v
RMS computation -> silence flag
        |
        v
XML speaker label assignment (dominant speaker per window)
        |
        v
[Model-specific embedding extraction]
        |
        v
Clustering / classification
        |
        v
Whisper transcription -> speaker-attributed transcript
```

### Segmentation parameters

| Experiment | Window | Hop | Overlap |
|---|---|---|---|
| Nada (CNN) | 3 s | 1.5 s | 50% |
| Nada (Transformer) | 30 s | 15 s | 50% |
| Maya/Steven/Hubert (HuBERT + Transformer) | 30 s | 15 s | 50% |
| Abdellah (HuBERT) | 30 s | 25 s | 17% |

### Silence detection

RMS energy is computed per window. Windows below the threshold are flagged as silent and excluded from training and evaluation.

| Experiment | RMS threshold |
|---|---|
| Nada | 0.00078 |
| Maya/Steven | 0.00061 |
| Abdellah | 0.005 |

### Speaker label assignment

For each audio window, the AMI segment XML files are parsed to compute the overlap between the window and each speaker's annotated segments. The speaker with the greatest overlap is assigned as the dominant label. Windows with no speaker overlap are labelled as silence.

---

## Experiments

---

### Nada — CNN from scratch + Transformer from scratch + Whisper

**Notebooks:**
- `CNN_Embeddings_Speaker_Diarization.ipynb`
- `Transformer_encoder_Diarization.ipynb`
- `Transformer_Whisper.ipynb`

**Dataset split:**

| Split | Meetings | Windows | Speakers |
|---|---|---|---|
| Train | EN2001a, EN2002a, EN2003a, EN2004a, EN2005a, EN2009b, IB4001 | 14,586 | 27 |
| Val | IN1001 | 2,220 | 3 |
| Test | IS1000a, TS3003a | 1,758 | 8 |

Total: 19,521 windows, 48 unique speakers, 957 silence windows.

**Stage 1 — CNN triplet encoder**

A four-layer CNN processes log-mel spectrograms (128 mel bands, 3 s windows) and outputs 256-dimensional L2-normalized embeddings. The model is trained with triplet margin loss (margin 0.3) using semi-hard negative mining and SpecAugment-style augmentation (time masking, frequency masking, gain jitter).

Architecture:

```
Conv2d(1->32) -> BN -> ReLU -> MaxPool
Conv2d(32->64) -> BN -> ReLU -> MaxPool
Conv2d(64->128) -> BN -> ReLU -> MaxPool
AdaptiveAvgPool -> FC(128->256->256->256) -> L2 normalize
```

Training configuration:

| Hyperparameter | Value |
|---|---|
| Epochs | 20 (early stopped at 6) |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Margin | 0.3 |
| Patience | 4 |
| Optimizer | Adam + weight decay 1e-4 |

Results:

| Metric | Value |
|---|---|
| Best val loss | 0.2796 |
| Val triplet accuracy | 58.11% |
| Test triplet accuracy | 55.92% |

**Stage 2 — Transformer encoder**

The trained CNN embeddings feed into a 2-layer Transformer encoder with positional embeddings. Sequences of 10 consecutive windows (30 s of context) are processed together. The model refines the per-window embeddings using self-attention and is trained with sequence-level triplet loss.

Architecture:

```
Linear(256->256) + PositionalEmbedding
TransformerEncoder(nhead=4, ff_dim=512, dropout=0.3, layers=2)
Linear(256->256) -> L2 normalize
```

Training configuration:

| Hyperparameter | Value |
|---|---|
| Epochs | 30 (early stopped at 19) |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Margin | 0.3 |
| Patience | 8 |
| Optimizer | Adam + weight decay 1e-4 |

Best val loss: 0.1863

**Stage 3 — Clustering and DER evaluation**

Embeddings are clustered with Agglomerative Clustering (cosine metric, average linkage). A sliding window average (window=5) smooths embeddings before clustering. DER is computed as the fraction of non-silence windows assigned to the wrong speaker after optimal Hungarian alignment.

| System | Train DER | Val DER | Test DER | Overall DER |
|---|---|---|---|---|
| CNN | 39.2% | 51.6% | 44.4% | 41.5% |
| Transformer | 49.4% | 56.7% | 67.2% | 53.7% |
| Transformer + Smoothing | 43.9% | 50.0% | 56.0% | 46.9% |

**Stage 4 — Transcription**

Whisper (`base`) transcribes each meeting's mixed audio with word-level timestamps. Words are aligned to speaker segments by matching each word's midpoint to the diarization timeline. Consecutive words from the same predicted speaker are merged into utterances.

Output: 2,101 utterances across all 10 meetings, saved as per-meeting CSV and TXT transcripts.

**Saved artifacts:**

| File | Description |
|---|---|
| `cnn_best.pth` | Best CNN encoder weights |
| `transformer_best.pth` | Best Transformer encoder weights |
| `cnn_embeddings_meta.csv` | Metadata for all 19,521 CNN embeddings |
| `diarization_output.csv` | Speaker segments (start, end, speaker, meeting) |
| `diarization_pipeline.pth` | Full pipeline config and Transformer weights |
| `all_transcripts.csv` | Speaker-attributed transcripts for all meetings |

**Plots:**

| Plot | Description |
|---|---|
| `cnn_loss.png` | CNN triplet loss train vs val |
| `transformer_loss.png` | Transformer triplet loss train vs val |
| `umap_transformer.png` | UMAP of Transformer embeddings per meeting |
| `der_comparison.png` | DER bar chart across all three systems |
| `cnn_vs_transformer_clusters.png` | Side-by-side UMAP for top 3 improved meetings |

---

### Maya, Steven — HuBERT + Transformer + Whisper

**Notebooks:** `HuBERT_Transformer_Diarization.ipynb`, `HuBERT_Whisper_Transcription.ipynb`

**Dataset:** 30 meetings (EN and ES series), 5,398 windows, 135 unique speaker IDs, 125 silence windows.

**Stage 1 — HuBERT feature extraction**

Raw 30-second audio windows are passed through `facebook/hubert-base-ls960` as a frozen feature extractor. Attention-mask-aware mean pooling is applied over the HuBERT frame sequence to produce one 768-dimensional L2-normalized embedding per window. HuBERT is not fine-tuned.

**Stage 2 — Temporal Transformer encoder**

The Transformer Encoder takes sequences of 8 consecutive HuBERT embeddings and refines them using self-attention. A learned positional embedding is added. The model is trained with a combined loss of CrossEntropy (class-balanced weights) and Supervised Contrastive Loss (lambda=0.3, temperature=0.1).

Architecture:

```
Linear(768->256) + LayerNorm + ReLU + Dropout
PositionalEmbedding(1, seq_len, 256)
TransformerEncoder(nhead=4, ff_dim=512, dropout=0.2, layers=2, norm_first=True, activation=gelu)
LayerNorm -> L2 normalize -> Linear(256->num_classes)
```

Training configuration:

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Patience | 6 |
| Lambda SupCon | 0.3 |
| Optimizer | AdamW + weight decay 1e-4 |

The train/val/test split is grouped by 5-minute blocks to prevent leakage across boundaries.

**Stage 3 — Evaluation**

Speaker classification accuracy is measured by training a Logistic Regression probe on the embeddings from each system on the same grouped test split. Clustering quality is measured with KMeans (k = number of true speakers) using ARI, NMI, Purity, and Silhouette.

**Stage 4 — Transcription**

Faster-Whisper (`small`) transcribes each 30-second audio window with VAD filtering enabled. The transcript is aligned to the predicted speaker timeline to produce a speaker-attributed output: `meeting + start_time + end_time + predicted_speaker_id + transcript`.

**Saved artifacts:**

| File | Description |
|---|---|
| `hubert_embeddings_768_direct_30s_15s_hop.npy` | Raw HuBERT embeddings (5,398 x 768) |
| `hubert_embeddings_768_direct_30s_15s_hop_metadata.csv` | Window metadata |
| `transformer_speaker_label_mapping.csv` | Speaker label mapping |
| `fair_hubert_vs_transformer_accuracy_comparison.csv` | Classification accuracy comparison |
| `hubert_vs_transformer_clustering_comparison.csv` | Clustering metrics (ARI, NMI, Purity, Silhouette) |
| `final_speaker_diarization_transcription.csv` | Speaker-attributed transcript |
| `predicted_speaker_timeline_transformer.csv` | Per-window prediction table |

---

### Abdellah — HuBERT + Whisper

**Notebook:** `HuBERT_Whisper_Diarization.ipynb`

**Dataset:** 10 meetings (same as Nada), 30 s windows, 25 s hop.

**Stage 1 — HuBERT feature extraction**

Raw 30-second windows are passed through `facebook/hubert-base-ls960` as a frozen feature extractor. Mean pooling over the frame sequence produces 768-dimensional embeddings.

**Stage 2 — KMeans clustering**

KMeans (k = number of true speakers per meeting, n_init=10) clusters the HuBERT embeddings. Hungarian algorithm alignment maps predicted cluster IDs to true speaker IDs and computes clustering accuracy.

**Stage 3 — Transcription**

Whisper (`openai/whisper-small`) transcribes each 30-second audio window via the Hugging Face pipeline. Transcripts are aligned to the per-window speaker predictions and saved as a speaker-attributed CSV and TXT.

**Saved artifacts:**

| File | Description |
|---|---|
| `hubert_speaker_segments.csv` | Per-window speaker and cluster labels |
| `speaker_aware_transcript.csv` | Speaker-attributed transcript (CSV) |
| `speaker_aware_transcript.txt` | Speaker-attributed transcript (plain text) |

---

### Youssef — WavLM + Whisper

**Notebook:** `WavLM_Whisper_Diarization.ipynb`

**Dataset:** AMI Meeting Corpus. Evaluated on meeting `TS3011a` (100 windows).

**Stage 1 — WavLM feature extraction**

Raw audio windows are passed through a WavLM-based feature extractor as a frozen pretrained model. Mean pooling over the frame sequence produces speaker embeddings per window.

**Stage 2 — Clustering**

KMeans clusters the WavLM embeddings. Hungarian algorithm alignment maps predicted cluster IDs to true speaker IDs.

**Stage 3 — Transcription and evaluation**

Whisper transcribes each audio window. Ground-truth word transcripts are extracted from the AMI manual word XML files (`ami_public_manual_1.6.2/words`). Each word's timestamp is matched to the corresponding audio window and used as the reference for WER/CER computation.

Results on `TS3011a` (100 windows evaluated):

| Metric | Value |
|---|---|
| Word Error Rate (WER) | 52.27% |
| Character Error Rate (CER) | 28.77% |
| Transcription Accuracy (1 - WER) | 47.73% |

**Saved artifacts:**

| File | Description |
|---|---|
| `hubert_speaker_segments.csv` | Per-window speaker and cluster labels |
| `speaker_aware_transcript.csv` | Speaker-attributed transcript (CSV) |
| `speaker_aware_transcript.txt` | Speaker-attributed transcript (plain text) |

---

## Repository Structure

```
.
├── nada/
│   ├── CNN_Embeddings_Speaker_Diarization.ipynb
│   ├── Transformer_encoder_Diarization.ipynb
│   └── Transformer_Whisper.ipynb
├── maya_steven/
│   ├── HuBERT_Transformer_Diarization.ipynb
│   └── HuBERT_Whisper_Transcription.ipynb
├── abdellah/
│   └── HuBERT_Whisper_Diarization.ipynb
├── youssef/
│   └── WavLM_Whisper_Diarization.ipynb
└── README.md
```

---

## Environment

Common dependencies:

```
torch
torchaudio
librosa
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
transformers
openai-whisper / faster-whisper
soundfile
h5py
umap-learn
jiwer
```

Nada's experiments run on Google Colab with GPU. Maya/Steven's experiments run on Google Colab with GPU. Abdellah's and Youssef's experiments run locally on Windows (CPU).

---

## Evaluation Metrics

| Metric | Used by | Description |
|---|---|---|
| DER | Nada | Fraction of non-silence windows assigned to the wrong speaker after Hungarian alignment |
| Triplet accuracy | Nada | Fraction of triplets where anchor is closer to positive than negative |
| Classification accuracy | Maya/Steven/Hubert | Logistic Regression probe on embeddings, grouped test split |
| ARI | Maya/Steven | Adjusted Rand Index between predicted clusters and true labels |
| NMI | Maya/Steven | Normalized Mutual Information |
| Purity | Maya/Steven | Fraction of windows assigned to the dominant class in each cluster |
| Silhouette | Maya/Steven | Internal cluster separation score |
| WER / CER | Youssef | Word/character error rate against AMI manual word XML transcripts |
| Clustering accuracy | Abdellah | Hungarian-aligned KMeans accuracy |

---

## Notes

- All experiments use the AMI manual segment XML files (`ami_public_manual_1.6.2`) for speaker label assignment. No ground-truth transcript is available, so ASR output is qualitative only.
- Silence windows are excluded from training and evaluation in all experiments.
- Speaker labels are local to each meeting; a global speaker ID is formed as `{meeting_id}.{speaker_letter}`.
- The Transformer in Nada's pipeline is trained with sequence-level triplet loss. The Transformer in Maya/Steven/Hubert's pipeline is trained with CrossEntropy and Supervised Contrastive Loss.
