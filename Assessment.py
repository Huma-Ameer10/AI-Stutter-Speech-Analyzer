import streamlit as st
import pandas as pd
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
#from streamlit_extras.colored_header import colored_header
import torch
import torchaudio
import time
from transformers import WhisperForAudioClassification, AutoFeatureExtractor
#from streamlit_option_menu import option_menu
#import matplotlib.colors as mcolors


# Set page title and favicon
st.set_page_config(page_title="Audio Visualization", page_icon="🎧")

# Upload audio file
audio_file = st.file_uploader("Upload Audio file for Assessment", type=["wav", "mp3"])

# Load the model and processor
model = WhisperForAudioClassification.from_pretrained("Huma10/Whisper_Stuttered_Speech")
feature_extractor = AutoFeatureExtractor.from_pretrained("Huma10/Whisper_Stuttered_Speech")
total_inference_time = 0  # Initialize the total inference time
# Check if an audio file is uploaded
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    # Load and preprocess the uploaded audio file
    input_audio, _ = torchaudio.load(audio_file)
    # Save the filename
    audio_filename = audio_file.name
    # Segment the audio into 3-second clips
    target_duration = 3  # 3 seconds
    target_samples = int(target_duration * 16000)
    num_clips = input_audio.size(1) // target_samples
    audio_clips = [input_audio[:, i * target_samples : (i + 1) * target_samples] for i in range(num_clips)]

    predicted_labels_list = []

    # Perform inference for each clip
    for clip in audio_clips:
        inputs = feature_extractor(clip.squeeze().numpy(), return_tensors="pt")
        input_features = inputs.input_features

        
        # Measure inference time
        start_time = time.time()
        # Perform inference
        with torch.no_grad():
            logits = model(input_features).logits

        end_time = time.time()
        inference_time = end_time - start_time
        total_inference_time += inference_time  # Accumulate inference time

        # Convert logits to predictions
        predicted_class_ids = torch.argmax(logits, dim=-1)
        predicted_labels = [model.config.id2label[class_id.item()] for class_id in predicted_class_ids]
        predicted_labels_list.extend(predicted_labels)
    
    st.markdown(f"Total inference time: **{total_inference_time:.4f}** seconds")
    def calculate_percentages(predicted_labels):
    # Count each type of disfluency
     disfluency_count = pd.Series(predicted_labels).value_counts(normalize=True)
     return disfluency_count * 100  # Convert fractions to percentages

    def plot_disfluency_percentages(percentages):
        fig, ax = plt.subplots()
        percentages.plot(kind='bar', ax=ax, color='#70bdbd')
        ax.set_title('Percentage of Each Disfluency Type')
        ax.set_xlabel('Disfluency Type')
        ax.set_ylabel('Percentage')
        plt.xticks(rotation=45)
        return fig

# Streamlit application
    def main():
        st.title("Speech Profile")
        st.write("This app analyzes the percentage of different types of disfluencies in stuttered speech.")

        # Calculate percentages
        percentages = calculate_percentages(predicted_labels_list)
        
        # Plot
        fig = plot_disfluency_percentages(percentages)
        st.pyplot(fig)


    main()

    success_check=st.success(' Assessment Completed Successfully!', icon="✅")
    time.sleep(5)
    success_check=st.empty()
        
    
 