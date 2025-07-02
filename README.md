Test the Project here: https://huggingface.co/spaces/somya-27-04-03/Song-vs-Speech-Classifier

---
title: "Song vs Speech Classifier"
emoji: 🎶
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.6.0"
app_file: app.py
pinned: false
---

# 🎶 Song vs Speech Classifier

This is a Gradio app that classifies uploaded audio files as either **Song** or **Speech** using a trained CNN model.

## 🚀 How to use

1. Click **Upload Audio**.
2. Select a `.wav` or compatible audio file.
3. The app will predict if it is **song** or **speech**.

## 📂 Files

- `app.py`: Main Gradio app file.
- `song_speech_cnn_model.h5`: Trained model.
- `requirements.txt`: Python dependencies.

## 🛠️ Model details

- Uses **MFCC feature extraction** via Librosa
- TensorFlow CNN backend for binary classification

---

### ✨ **Demo**

Try it out by uploading any clear audio snippet (singing vs speech).

---

> created by Somya Ranjan Mahapatra
