# 🎧 AI-Stutter-Speech-Analyzer

This **Streamlit application** is designed to analyze stuttered speech by detecting and visualizing various disfluency types. It leverages **Hugging Face Transformers**, **PyTorch**, and **Streamlit** for real-time audio analysis and visualization.

---

## 🚀 Features

- **Upload Audio Files**:
  - Supports `.wav` and `.mp3` formats.
  - Provides audio playback within the app.

- **Audio Processing**:
  - Segments audio files into 3-second clips for efficient analysis.
  - Performs inference using a fine-tuned model for stuttered speech classification.

- **Real-Time Results**:
  - Displays the total inference time.
  - Generates a bar chart showing the percentage distribution of disfluency types.

- **Interactive Visualization**:
  - Provides an easy-to-understand visual breakdown of disfluency types in speech.

---

## 📂 File Structure

- **`Assessment.py`**: Main Streamlit app containing the audio processing and visualization logic.
- **Model and Processor**:
  - `WhisperForAudioClassification`: A fine-tuned Hugging Face model for speech classification.
  - `AutoFeatureExtractor`: Preprocessor for feature extraction from audio inputs.

---

## 🛠️ Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
