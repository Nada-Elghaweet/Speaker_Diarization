"""
Streamlit GUI: Upload a meeting audio file and run an end-to-end
Speaker Diarization + Transcription pipeline.

Pipeline:
1) Upload one mixed meeting audio file OR multiple headset/channel audio files.
2) Load audio at 16 kHz.
3) Split audio into 30-second windows with 15-second hop.
4) Extract HuBERT embeddings for each non-silent window.
5) Cluster embeddings into Speaker 1, Speaker 2, ... using KMeans.
6) Transcribe each window using Faster-Whisper.
7) Export the speaker-attributed transcript as CSV.

Notes:
- This version is designed for a new uploaded meeting where true speaker labels are unknown.
- Therefore, it uses clustering to produce Speaker 1 / Speaker 2 / ... labels.
- Your notebook's supervised Transformer classifier is trained on AMI labels, so it is not ideal
  for arbitrary new meetings unless you save and load the exact trained artifacts and use a compatible dataset.
"""

from __future__ import annotations

import io
import math
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# Heavy imports are kept inside cached functions where possible.

TARGET_SR = 16000
DEFAULT_WINDOW_SEC = 30.0
DEFAULT_HOP_SEC = 15.0
DEFAULT_RMS_THRESHOLD = 0.00061
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"


st.set_page_config(
    page_title="Meeting Diarization + Transcription",
    page_icon="🎙️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def seconds_to_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    if seconds is None or np.isnan(seconds):
        return "00:00:00.000"
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio safely to [-1, 1]."""
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 1e-8:
        audio = audio / max_abs
    return audio.astype(np.float32)


def compute_rms(audio: np.ndarray) -> float:
    """Root Mean Square energy."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2)))


def save_uploaded_file_to_temp(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temporary path and return the path."""
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_one_audio(uploaded_file, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load one uploaded audio file with librosa."""
    import librosa

    tmp_path = save_uploaded_file_to_temp(uploaded_file)
    try:
        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=True)
    finally:
        # Keep cleanup simple; Windows sometimes locks files during decode.
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass

    return normalize_audio(audio), target_sr


def load_and_mix_audio(uploaded_files: List, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int, List[str]]:
    """
    Load a single meeting file or multiple channel/headset files.
    If multiple files are uploaded, trim to the shortest length and average them.
    """
    if not uploaded_files:
        raise ValueError("No audio file was uploaded.")

    audios = []
    names = []
    for file in uploaded_files:
        audio, sr = load_one_audio(file, target_sr=target_sr)
        if len(audio) == 0:
            continue
        audios.append(audio)
        names.append(file.name)

    if not audios:
        raise ValueError("Could not load any audio from the uploaded file(s).")

    if len(audios) == 1:
        return audios[0], target_sr, names

    min_len = min(len(a) for a in audios)
    trimmed = [a[:min_len] for a in audios]
    mixed = np.mean(np.stack(trimmed, axis=0), axis=0)
    return normalize_audio(mixed), target_sr, names


def sliding_windows(
    audio: np.ndarray,
    sr: int,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
) -> List[dict]:
    """Split audio into overlapping windows. Keeps final tail by adding a final window."""
    window_size = int(window_sec * sr)
    hop_size = int(hop_sec * sr)

    if window_size <= 0:
        raise ValueError("window_sec must be positive.")
    if hop_size <= 0:
        raise ValueError("hop_sec must be positive.")
    if hop_size > window_size:
        raise ValueError("hop_sec should be <= window_sec.")

    n = len(audio)
    windows: List[dict] = []

    if n == 0:
        return windows

    if n < window_size:
        padded = np.pad(audio, (0, window_size - n), mode="constant")
        windows.append(
            {
                "window_idx": 0,
                "start_time": 0.0,
                "end_time": n / sr,
                "audio": padded.astype(np.float32),
                "is_padded": True,
            }
        )
        return windows

    idx = 0
    for start in range(0, n - window_size + 1, hop_size):
        end = start + window_size
        windows.append(
            {
                "window_idx": idx,
                "start_time": start / sr,
                "end_time": end / sr,
                "audio": audio[start:end].astype(np.float32),
                "is_padded": False,
            }
        )
        idx += 1

    last_end_sample = int(windows[-1]["end_time"] * sr)
    if last_end_sample < n:
        start = n - window_size
        end = n
        if start > int(windows[-1]["start_time"] * sr):
            windows.append(
                {
                    "window_idx": idx,
                    "start_time": start / sr,
                    "end_time": end / sr,
                    "audio": audio[start:end].astype(np.float32),
                    "is_padded": False,
                }
            )

    return windows


# -----------------------------------------------------------------------------
# Cached ML model loaders
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_hubert(device_choice: str = "auto"):
    """Load HuBERT and its feature extractor."""
    import torch
    from transformers import HubertModel, Wav2Vec2FeatureExtractor

    if device_choice == "cuda" and not torch.cuda.is_available():
        device_choice = "cpu"

    device = torch.device("cuda" if (device_choice == "auto" and torch.cuda.is_available()) else device_choice)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
    hubert = HubertModel.from_pretrained(HUBERT_MODEL_NAME).to(device)
    hubert.eval()

    for param in hubert.parameters():
        param.requires_grad = False

    return feature_extractor, hubert, device


@st.cache_resource(show_spinner=False)
def load_whisper(model_size: str = "small", device_choice: str = "auto"):
    """Load Faster-Whisper model."""
    import torch
    from faster_whisper import WhisperModel

    if device_choice == "cuda" and not torch.cuda.is_available():
        device_choice = "cpu"

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model, device, compute_type


# -----------------------------------------------------------------------------
# Feature extraction, clustering, transcription
# -----------------------------------------------------------------------------

def masked_mean_pool(hubert_model, hidden, attention_mask=None):
    """Mean-pool HuBERT frames while ignoring padded waveform regions."""
    import torch

    if attention_mask is None:
        return hidden.mean(dim=1)

    if hasattr(hubert_model, "_get_feature_vector_attention_mask"):
        frame_mask = hubert_model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
    else:
        frame_mask = torch.nn.functional.interpolate(
            attention_mask[:, None].float(), size=hidden.shape[1], mode="nearest"
        ).squeeze(1).bool()

    frame_mask = frame_mask.to(hidden.device).unsqueeze(-1).type_as(hidden)
    summed = (hidden * frame_mask).sum(dim=1)
    counts = frame_mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


def extract_hubert_embeddings(
    windows: List[dict],
    batch_size: int,
    rms_threshold: float,
    device_choice: str,
) -> Tuple[np.ndarray, List[int]]:
    """Extract 768-dimensional L2-normalized HuBERT embeddings for non-silent windows."""
    import torch
    import torch.nn.functional as F

    non_silent_indices = [i for i, w in enumerate(windows) if compute_rms(w["audio"]) >= rms_threshold]
    if not non_silent_indices:
        return np.empty((0, 768), dtype=np.float32), []

    feature_extractor, hubert, device = load_hubert(device_choice=device_choice)

    embeddings = []
    progress = st.progress(0, text="Extracting HuBERT embeddings...")

    for start in range(0, len(non_silent_indices), batch_size):
        batch_indices = non_silent_indices[start : start + batch_size]
        batch_audio = [windows[i]["audio"] for i in batch_indices]

        inputs = feature_extractor(
            batch_audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )

        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

        with torch.no_grad():
            outputs = hubert(input_values=input_values, attention_mask=attention_mask)
            pooled = masked_mean_pool(hubert, outputs.last_hidden_state, attention_mask)
            pooled = F.normalize(pooled, p=2, dim=1)

        embeddings.append(pooled.cpu().numpy().astype(np.float32))
        progress.progress(
            min((start + len(batch_indices)) / len(non_silent_indices), 1.0),
            text=f"Extracting HuBERT embeddings... {start + len(batch_indices)}/{len(non_silent_indices)}",
        )

    progress.empty()
    return np.vstack(embeddings), non_silent_indices


def choose_auto_speakers(embeddings: np.ndarray, max_speakers: int = 6) -> int:
    """Estimate number of speakers using silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    n = len(embeddings)
    if n < 2:
        return 1

    max_k = min(max_speakers, n - 1)
    if max_k < 2:
        return 1

    X = StandardScaler().fit_transform(embeddings)
    best_k = 2
    best_score = -1.0

    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return int(best_k)


def cluster_speakers(
    embeddings: np.ndarray,
    non_silent_indices: List[int],
    total_windows: int,
    speaker_setting: str,
    max_auto_speakers: int = 6,
) -> Tuple[List[str], int]:
    """Cluster non-silent windows into speakers and return speaker labels per window."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    labels_per_window = ["silence" for _ in range(total_windows)]

    if len(non_silent_indices) == 0:
        return labels_per_window, 0

    if speaker_setting == "Auto":
        n_speakers = choose_auto_speakers(embeddings, max_speakers=max_auto_speakers)
    else:
        n_speakers = int(speaker_setting)

    n_speakers = max(1, min(n_speakers, len(non_silent_indices)))

    if n_speakers == 1:
        cluster_ids = np.zeros(len(non_silent_indices), dtype=int)
    else:
        X = StandardScaler().fit_transform(embeddings)
        cluster_ids = KMeans(n_clusters=n_speakers, random_state=42, n_init=10).fit_predict(X)

    # Rename clusters by first appearance in the meeting, not arbitrary KMeans ID.
    cluster_to_speaker = {}
    next_id = 1
    for original_window_idx, cluster_id in zip(non_silent_indices, cluster_ids):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_to_speaker:
            cluster_to_speaker[cluster_id] = f"Speaker {next_id}"
            next_id += 1
        labels_per_window[original_window_idx] = cluster_to_speaker[cluster_id]

    return labels_per_window, n_speakers


def transcribe_windows(
    windows: List[dict],
    speaker_labels: List[str],
    rms_threshold: float,
    whisper_model_size: str,
    language_choice: str,
    device_choice: str,
    transcribe_silence: bool = False,
) -> List[str]:
    """Transcribe each audio window using Faster-Whisper."""
    whisper_model, device, compute_type = load_whisper(whisper_model_size, device_choice=device_choice)

    language = None if language_choice == "auto" else language_choice
    transcripts: List[str] = []

    progress = st.progress(0, text="Transcribing windows with Faster-Whisper...")
    for i, w in enumerate(windows):
        rms = compute_rms(w["audio"])
        if (speaker_labels[i] == "silence" or rms < rms_threshold) and not transcribe_silence:
            transcripts.append("")
        else:
            try:
                segments, _info = whisper_model.transcribe(
                    w["audio"].astype(np.float32),
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                transcripts.append(text)
            except Exception as exc:
                transcripts.append(f"[TRANSCRIPTION_ERROR: {exc}]")

        progress.progress(
            min((i + 1) / len(windows), 1.0),
            text=f"Transcribing windows... {i + 1}/{len(windows)}",
        )

    progress.empty()
    return transcripts


def build_results_dataframe(
    windows: List[dict],
    speaker_labels: List[str],
    transcripts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build final output DataFrame."""
    if transcripts is None:
        transcripts = [""] * len(windows)

    rows = []
    for w, speaker, text in zip(windows, speaker_labels, transcripts):
        start = float(w["start_time"])
        end = float(w["end_time"])
        rows.append(
            {
                "window_idx": int(w["window_idx"]),
                "start_time": start,
                "end_time": end,
                "start_hms": seconds_to_time(start),
                "end_hms": seconds_to_time(end),
                "speaker": speaker,
                "rms": compute_rms(w["audio"]),
                "is_padded": bool(w.get("is_padded", False)),
                "transcript": text,
            }
        )

    return pd.DataFrame(rows)


def merge_consecutive_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive rows with the same speaker, combining transcript text."""
    if df.empty:
        return df

    merged = []
    current = None

    for _, row in df.iterrows():
        speaker = row["speaker"]
        text = str(row.get("transcript", "") or "").strip()

        if current is None:
            current = row.to_dict()
            current["transcript"] = text
            continue

        same_speaker = speaker == current["speaker"]
        if same_speaker:
            current["end_time"] = row["end_time"]
            current["end_hms"] = row["end_hms"]
            if text:
                current["transcript"] = (str(current.get("transcript", "")).strip() + " " + text).strip()
        else:
            merged.append(current)
            current = row.to_dict()
            current["transcript"] = text

    if current is not None:
        merged.append(current)

    out = pd.DataFrame(merged)
    cols = ["start_time", "end_time", "start_hms", "end_hms", "speaker", "transcript"]
    return out[cols]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("🎙️ Meeting Speaker Diarization + Transcription")
st.caption("Upload a meeting audio file and get: who spoke, when, and what was said.")

with st.sidebar:
    st.header("Settings")

    st.subheader("Windowing")
    window_sec = st.number_input("Window length seconds", min_value=5.0, max_value=60.0, value=30.0, step=5.0)
    hop_sec = st.number_input("Hop seconds", min_value=1.0, max_value=60.0, value=15.0, step=1.0)
    rms_threshold = st.number_input(
        "Silence RMS threshold",
        min_value=0.0,
        max_value=0.1,
        value=float(DEFAULT_RMS_THRESHOLD),
        format="%.6f",
        step=0.0001,
    )

    st.subheader("Speakers")
    speaker_setting = st.selectbox("Number of speakers", ["Auto", "1", "2", "3", "4", "5", "6", "7", "8"], index=2)
    max_auto_speakers = st.slider("Max speakers for Auto", min_value=2, max_value=10, value=6)

    st.subheader("Models")
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    hubert_batch_size = st.slider("HuBERT batch size", min_value=1, max_value=8, value=2)
    whisper_model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=2)
    language_choice = st.selectbox("Transcription language", ["auto", "en", "ar"], index=1)
    run_transcription = st.checkbox("Run transcription", value=True)

    st.info(
        "First run may take time because HuBERT and Whisper models may download/load. "
        "For weak laptops, choose Whisper tiny/base and CPU."
    )

uploaded_files = st.file_uploader(
    "Upload meeting audio file(s)",
    type=["wav", "mp3", "m4a", "flac", "ogg"],
    accept_multiple_files=True,
    help="Upload one mixed meeting file, or upload multiple headset/channel files and the app will mix them.",
)

run_button = st.button("🚀 Process Meeting", type="primary", disabled=not uploaded_files)

if not uploaded_files:
    st.markdown(
        """
        ### How this app works
        1. Upload one meeting audio file, for example `meeting.wav`.
        2. Or upload multiple headset/channel files, and the app will mix them first.
        3. Click **Process Meeting**.
        4. The app outputs a speaker timeline and a downloadable CSV transcript.

        **Output speakers are anonymous:** `Speaker 1`, `Speaker 2`, etc. because a new uploaded meeting has no ground-truth names.
        """
    )

if run_button and uploaded_files:
    try:
        if hop_sec > window_sec:
            st.error("Hop seconds must be smaller than or equal to window length.")
            st.stop()

        with st.status("Loading uploaded audio...", expanded=True) as status:
            audio, sr, names = load_and_mix_audio(uploaded_files, target_sr=TARGET_SR)
            duration_sec = len(audio) / sr
            status.write(f"Loaded file(s): {', '.join(names)}")
            status.write(f"Sample rate: {sr} Hz")
            status.write(f"Duration: {seconds_to_time(duration_sec)}")
            status.update(label="Audio loaded successfully", state="complete")

        col1, col2, col3 = st.columns(3)
        col1.metric("Duration", seconds_to_time(duration_sec))
        col2.metric("Sample rate", f"{sr} Hz")
        col3.metric("Uploaded files", len(uploaded_files))

        st.audio(audio, sample_rate=sr)

        with st.status("Splitting audio into windows...", expanded=False) as status:
            windows = sliding_windows(audio, sr, window_sec=window_sec, hop_sec=hop_sec)
            if not windows:
                st.error("No windows were produced from this audio.")
                st.stop()
            status.write(f"Total windows: {len(windows)}")
            status.update(label="Windowing complete", state="complete")

        with st.status("Running HuBERT speaker embeddings + clustering...", expanded=True) as status:
            embeddings, non_silent_indices = extract_hubert_embeddings(
                windows=windows,
                batch_size=hubert_batch_size,
                rms_threshold=rms_threshold,
                device_choice=device_choice,
            )
            status.write(f"Non-silent windows: {len(non_silent_indices)} / {len(windows)}")

            speaker_labels, estimated_speakers = cluster_speakers(
                embeddings=embeddings,
                non_silent_indices=non_silent_indices,
                total_windows=len(windows),
                speaker_setting=speaker_setting,
                max_auto_speakers=max_auto_speakers,
            )
            status.write(f"Speakers used: {estimated_speakers}")
            status.update(label="Speaker diarization complete", state="complete")

        transcripts = None
        if run_transcription:
            with st.status("Running Faster-Whisper transcription...", expanded=True) as status:
                transcripts = transcribe_windows(
                    windows=windows,
                    speaker_labels=speaker_labels,
                    rms_threshold=rms_threshold,
                    whisper_model_size=whisper_model_size,
                    language_choice=language_choice,
                    device_choice=device_choice,
                )
                status.update(label="Transcription complete", state="complete")

        results_df = build_results_dataframe(windows, speaker_labels, transcripts)
        merged_df = merge_consecutive_segments(results_df)

        st.success("Meeting processed successfully.")

        tab1, tab2, tab3 = st.tabs(["Final transcript", "Window timeline", "Downloads"])

        with tab1:
            st.subheader("Speaker-attributed transcript")
            st.dataframe(merged_df, use_container_width=True, hide_index=True)

            for _, row in merged_df.iterrows():
                speaker = row["speaker"]
                time_range = f'{row["start_hms"]} → {row["end_hms"]}'
                text = str(row.get("transcript", "") or "").strip()
                if not text:
                    text = "[No transcript / silence]"
                st.markdown(f"**{speaker}** · `{time_range}`  ")
                st.write(text)
                st.divider()

        with tab2:
            st.subheader("Detailed window-level timeline")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

        with tab3:
            csv_window = results_df.to_csv(index=False).encode("utf-8-sig")
            csv_merged = merged_df.to_csv(index=False).encode("utf-8-sig")

            st.download_button(
                "Download window-level CSV",
                data=csv_window,
                file_name="meeting_window_speaker_transcript.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download merged transcript CSV",
                data=csv_merged,
                file_name="meeting_merged_speaker_transcript.csv",
                mime="text/csv",
            )

            st.caption("UTF-8-SIG is used so the CSV opens correctly in Excel.")

    except Exception as exc:
        st.error("Something went wrong while processing the meeting.")
        st.exception(exc)
        st.markdown(
            """
            Common fixes:
            - Use a `.wav` file if `.mp3`/`.m4a` fails.
            - Install FFmpeg if compressed audio decoding fails.
            - Use Whisper `tiny` or `base` if your laptop is slow.
            - Reduce HuBERT batch size to `1` if memory is low.
            """
        )
