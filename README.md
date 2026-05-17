<div align="center">

# Speaker Diarization Pipeline
### WavLM + Whisper + Temporal Transformer

*An end-to-end notebook for answering "who spoke when?" on the AMI Meeting Corpus*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![AMI Corpus](https://img.shields.io/badge/Dataset-AMI_Corpus-00A878?style=for-the-badge)](https://groups.inf.ed.ac.uk/ami/corpus/)

</div>

---

## What This Notebook Does

This notebook walks through a complete speaker diarization pipeline — from raw WAV files all the way to a timestamped, speaker-labeled transcript with Word Error Rate evaluation. Everything runs in a single notebook, sequentially, and every intermediate result is saved to disk so you can pick up from any stage without re-running everything from scratch.

The core idea is to combine two pre-trained models — **WavLM** for acoustic speaker identity and **Whisper** for linguistic context — into a single 1280-dimensional feature vector per audio window, then train a lightweight Transformer on top to add temporal awareness across consecutive windows.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Audio Preprocessing](#audio-preprocessing)
- [Speaker Label Assignment](#speaker-label-assignment)
- [Feature Extraction](#feature-extraction)
- [Data Splits](#data-splits)
- [Baseline Classifier](#baseline-classifier)
- [Temporal Transformer](#temporal-transformer)
- [Unsupervised Diarization](#unsupervised-diarization)
- [Transcription](#transcription)
- [WER / CER Evaluation](#wer--cer-evaluation)
- [Model Saving](#model-saving)
- [Outputs](#outputs)
- [Requirements](#requirements)

---

## Environment Setup

The first thing the notebook does is redirect all model caches away from the system drive. WavLM and Whisper together can be several gigabytes, so before anything is imported or downloaded, the cache directories for Hugging Face and PyTorch are pointed at a dedicated folder on the D drive.

```python
os.environ["HF_HOME"]    = r"D:\ANN_Project_Cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\ANN_Project_Cache\torch"
```

If you are running this on a machine where C drive space is not a concern, you can remove these lines and everything will download to the default `~/.cache` locations.

Four path constants are then defined and used throughout the rest of the notebook:

| Variable | Points To |
|:---|:---|
| `DATA_ROOT` | Root of the AMI corpus — each subfolder is a meeting |
| `MANUAL_PATH` | AMI manual annotations — XML files for ground truth |
| `OUTPUT_ROOT` | Where intermediate files are saved (audio chunks, metadata CSVs) |
| `SAVE_DIR` | Where final artifacts are saved (embeddings, model weights, transcripts) |

---

## Dataset

The notebook uses the **AMI Meeting Corpus** — a collection of recorded multi-party meetings, each with separate headset microphone tracks per speaker and detailed manual annotations.

| Property | Value |
|:---|:---|
| Meetings used | 10 (configurable via `NUM_MEETINGS`) |
| Speakers per meeting | 4 to 5 |
| Audio sample rate | 16 kHz |
| Annotation format | XML — segment-level and word-level |
| Random seed | 42 |

The notebook randomly samples `NUM_MEETINGS` meetings from whatever is available under `DATA_ROOT`, then does a rough 80/20 meeting-level split. The actual train/val/test splits on embeddings happen later using group-aware splitting.

Setting `NUM_MEETINGS = None` will use every meeting found in the corpus.

---

## Audio Preprocessing

### Loading and Mixing

Each AMI meeting contains separate WAV files per speaker — one per headset microphone, named `Headset-0` through `Headset-4`. The notebook loads all available headset tracks, truncates them to the same length (the shortest track), sums them into a single mono stream, and normalizes the result to the range [-1, 1].

```python
mixed = np.sum(list(tracks.values()), axis=0)
mixed = mixed / (np.max(np.abs(mixed)) + 1e-8)
```

The small constant `1e-8` prevents division by zero on silent recordings.

### Sliding Windows

Audio is segmented using a sliding window approach:

| Parameter | Value | Reasoning |
|:---|:---|:---|
| Window length | 30 seconds | Matches Whisper's native input size; long enough to capture speaker patterns |
| Hop size | 15 seconds | 50% overlap ensures each speaker transition is captured in at least two windows |
| Overlap | 15 seconds | Gives the Transformer temporal continuity across adjacent windows |

If a recording is shorter than one window, it is zero-padded to reach 30 seconds and flagged as `is_padded`. The function also handles the tail end of audio — if the last regular window does not reach the end of the recording, one final window is anchored at the last 30 seconds to avoid leaving audio uncovered.

### Silence Detection

Each window's RMS energy is computed and compared against a fixed threshold:

```python
RMS_SILENCE_THRESHOLD = 0.00061
```

Windows below this threshold are flagged as silent and excluded from training. This value was calibrated for the AMI corpus — raising it discards more windows including quiet speech, lowering it lets more noise through.

A waveform plot is generated for the first meeting so you can visually verify that the threshold sits at a reasonable level before processing everything.

### Saved Output

Every window is saved as an individual `.npy` file. A metadata CSV is written alongside it with one row per window:

```
meeting, window_idx, start_time, end_time, rms, is_silent, audio_path
```

---

## Speaker Label Assignment

Ground-truth speaker labels are extracted from the AMI manual annotation XML files. Each XML file covers one speaker in one meeting and lists the time intervals during which that person was speaking.

For each 30-second audio window, the notebook calculates how many seconds each speaker actually spoke within that window and assigns the label of whichever speaker had the most total speech time — the "dominant speaker." Windows where no speaker is active are labeled as silence.

```python
overlap += min(window_end, seg_end) - max(window_start, seg_start)
```

This intersection formula correctly handles partial overlaps at the edges of windows.

All speaker IDs (string format like `EN2002a.MEE071`) are converted to integers using `sklearn.LabelEncoder` and the mapping is saved so predictions can be converted back to readable IDs later.

---

## Feature Extraction

### Why Two Models?

WavLM and Whisper are complementary. WavLM was pre-trained specifically to model speaker identity — its representations are sensitive to *how* someone sounds. Whisper's encoder was trained on speech-to-text and picks up on *what* is being said and the linguistic patterns associated with it. Concatenating them gives the downstream model more information to work with than either one alone.

### WavLM — 768 Dimensions

**Model:** `microsoft/wavlm-base-plus`

WavLM processes the raw waveform and produces a sequence of hidden state vectors — one per roughly 20ms of audio. These are aggregated into a single 768-dimensional vector using masked average pooling: only frames corresponding to real audio (not padding) are included in the average. The result is L2-normalized.

Both models are loaded in inference mode with gradients disabled — they are used purely as feature extractors and their weights are not updated during training.

```python
wavlm.eval()
for p in wavlm.parameters():
    p.requires_grad = False
```

### Whisper — 512 Dimensions

**Model:** `openai/whisper-base` (encoder only, no decoder)

Each audio window is padded or trimmed to exactly 30 seconds, converted to a log-mel spectrogram, and passed through the Whisper encoder. The output sequence is mean-pooled across time to produce a 512-dimensional vector.

### Combined Embedding — 1280 Dimensions

```python
combined = np.concatenate((wavlm_embs, whisp_embs), axis=1)
```

The two vectors are concatenated, giving a 1280-dimensional representation per window. These are processed in batches of 8 on GPU or 2 on CPU and saved as a single `.npy` matrix alongside the metadata CSV.

---

## Data Splits

After loading the saved embeddings, three filtering steps are applied before splitting:

1. **Silence removal** — windows flagged as silent are dropped
2. **Rare class removal** — any speaker with fewer than 5 windows is removed; a classifier cannot learn meaningfully from 1 or 2 examples
3. **Temporal ordering** — all windows are sorted chronologically within each meeting before splitting, which is required for the sequence-based Transformer to work correctly

Labels are then remapped to contiguous integers starting at 0, since filtering may have created gaps in the label sequence.

### Group-Aware Splitting

A standard random split would be incorrect here. Windows from the same 5-minute block of the same meeting are highly correlated — a model could memorize speaker patterns from the first half of a meeting and score well on the second half without generalizing at all. To prevent this, windows are grouped into 5-minute blocks per meeting and `GroupShuffleSplit` is used to ensure no group appears on both sides of any split.

```
Split           Proportion
Train             ~64%
Validation        ~16%
Test              ~20%
```

After splitting, validation and test sets are further filtered to only include speakers that also appear in the training set.

---

## Baseline Classifier

Before training the Transformer, a logistic regression is fitted directly on the raw 1280-d embeddings. This tells you how much speaker information is already present in the features without any temporal modeling — the Transformer should meaningfully exceed this number.

```python
make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
)
```

`class_weight="balanced"` adjusts for speakers with fewer windows. `max_iter=5000` is necessary because convergence with 1280 features and multiple classes takes longer than the scikit-learn default of 100 iterations.

A 3D PCA scatter plot is also generated at this stage to visually inspect whether speakers form separable clusters in the embedding space before any neural model is involved.

---

## Temporal Transformer

### Sequence Dataset

Audio windows are grouped into sequences of 5 consecutive windows. Each sequence covers roughly 2.5 minutes of audio with 50% overlap between adjacent windows. The label for each sequence is the speaker of the **last** window — the model learns to predict who is speaking now given recent context.

```python
SEQUENCE_LENGTH = 5
```

### Model Architecture

```
Input:  [batch, 5, 1280]
         |
         Linear(1280 -> 256)          # project to smaller internal dimension
         |
         TransformerEncoder
           2 layers
           4 attention heads
           feedforward dim: 512
           dropout: 0.3
         |
         Mean pool across sequence dimension
         |
         Linear(256 -> num_classes)
Output: [batch, num_classes]
```

The model has a dual-mode forward pass controlled by an `extract_embeddings` flag. When set to `True`, it returns the 256-dimensional pooled representation for use in K-Means clustering. When `False`, it returns class logits for supervised training and evaluation.

### Training Configuration

| Setting | Value |
|:---|:---|
| Loss | CrossEntropyLoss |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 64 |
| Epochs | 30 |

Training and validation accuracy are tracked per epoch and plotted at the end. A healthy training run shows both curves rising together — a large gap between train and validation accuracy is a sign of overfitting.

---

## Unsupervised Diarization

After supervised training, the Transformer's internal representations are used for open-set diarization — the more realistic scenario where speaker identities are not known in advance.

The Transformer is run over the test set with `extract_embeddings=True` to collect 256-dimensional vectors. K-Means clustering is then run **independently per meeting**, using the ground-truth number of speakers as K. This simulates a real deployment where the audio belongs to one session and the goal is to separate the speakers within it, without any predefined identity labels.

Two metrics are reported per meeting and averaged:

**Adjusted Rand Index (ARI)** measures agreement between predicted clusters and ground truth, adjusted for chance. It ranges from -1 to 1 — higher is better, 0 means the clustering is no better than random.

**Silhouette Score** measures how compact and well-separated the clusters are without using labels at all. It ranges from -1 to 1 — higher means cleaner separation.

A side-by-side 2D PCA scatter plot shows ground-truth speaker distributions next to K-Means cluster assignments, giving a visual sense of how well the learned representations support unsupervised separation.

---

## Transcription

Each test window is transcribed using **faster-whisper** — a CTranslate2-optimized reimplementation of Whisper that runs 2 to 4 times faster than the original library with no change in output quality.

```python
WhisperModel("small", device=device, compute_type="float16")  # GPU
WhisperModel("small", device=device, compute_type="int8")     # CPU
```

The `vad_filter=True` option applies Voice Activity Detection internally, skipping silent regions and reducing hallucinated text on quiet windows.

The final output is a table with one row per audio window:

| Column | Contents |
|:---|:---|
| `meeting` | Meeting identifier |
| `start_time` / `end_time` | Window boundaries in seconds |
| `true_speaker_id` | Ground-truth speaker from XML annotations |
| `predicted_supervised_id` | Transformer classifier prediction |
| `predicted_speaker_cluster` | K-Means cluster assignment |
| `transcript` | Whisper-generated text for this window |

This table is saved as `final_speaker_diarization_transcription.csv`.

---

## WER / CER Evaluation

Transcription quality is evaluated against the AMI word-level XML annotations. These files contain every word spoken in a meeting along with precise start and end timestamps.

For each 30-second window, the notebook collects all ground-truth words whose timestamps fall within that window, joins them into a reference string, and compares it to the Whisper output using the `jiwer` library.

```
Word Error Rate (WER)       =  (Substitutions + Insertions + Deletions) / Total Reference Words
Character Error Rate (CER)  =  same formula applied at the character level
```

Three values are printed per evaluation:

- **WER** — overall word-level error rate
- **CER** — character-level error rate, less affected by tokenization differences
- **Linguistic Accuracy** — `max(0, 1 - WER)`, floored at zero since WER can exceed 100% when many extra words are inserted

A short sample comparison of ground-truth text versus Whisper output is printed for manual inspection.

To evaluate a different meeting, change `TARGET_MEETING` at the top of the cell to any meeting ID present in the test split.

---

## Model Saving

All artifacts needed to reproduce inference are saved under `SAVE_DIR/web_models/`:

| File | Contents |
|:---|:---|
| `temporal_transformer.pt` | Transformer state dict |
| `wavlm_encoder.pt` | WavLM encoder state dict |
| `feature_extractor/` | Hugging Face feature extractor config and vocabulary |
| `baseline_clf.joblib` | Scikit-learn pipeline — StandardScaler and LogisticRegression |
| `speaker_label_mapping.csv` | Mapping from integer labels back to speaker ID strings |

To reload the Transformer for inference:

```python
model = SpeakerTransformerRefiner(num_train_classes=N)
model.load_state_dict(torch.load("web_models/temporal_transformer.pt"))
model.eval()
```

The Whisper model used for transcription is not saved here — faster-whisper downloads and caches it automatically on first use.

---

## Outputs

| File | Description |
|:---|:---|
| `windows_audio_30s_15s_hop/` | Individual `.npy` audio chunks, one per window |
| `windows_metadata.csv` | Audio window index with RMS values and silence flags |
| `labeled_metadata.csv` | Same index with dominant speaker labels added |
| `speaker_classes.csv` | Ordered list of speaker ID strings matching integer labels |
| `wavlm_whisper_embeddings_1280.npy` | Full embedding matrix, one 1280-d row per window |
| `wavlm_whisper_embeddings_metadata.csv` | Metadata aligned row-for-row to the embedding matrix |
| `final_speaker_diarization_transcription.csv` | Diarized transcript with ground truth and predictions |
| `web_models/` | All saved model artifacts |

---

## Requirements

```
torch >= 2.0
torchaudio
transformers
openai-whisper
faster-whisper
librosa
soundfile
numpy
pandas
scikit-learn
matplotlib
tqdm
jiwer
joblib
```

Install with:

```bash
pip install torch torchaudio transformers openai-whisper faster-whisper \
            librosa soundfile numpy pandas scikit-learn matplotlib tqdm \
            jiwer joblib
```

---

## Notes

- The notebook is designed to run top-to-bottom. Each cell saves its outputs, so if a later cell fails you can reload from disk without re-running the expensive feature extraction step.
- `BATCH_SIZE` is automatically set to 8 on GPU and 2 on CPU. On a CPU-only machine, the embedding extraction step will take a while — expect several minutes per meeting.
- The silence threshold (`0.00061`) and sequence length (`5`) were chosen for the AMI corpus and may need adjustment on other datasets.
- `GroupShuffleSplit` is used deliberately throughout. A standard random split leaks information between temporally adjacent windows and produces accuracy numbers that look better than they actually are.

---

<div align="center">

ANN Project — Speaker Diarization

[github.com/Nada-Elghaweet/Speaker_Diarization](https://github.com/Nada-Elghaweet/Speaker_Diarization)

</div>
