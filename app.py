# app.py
import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
# Load the trained model
model = tf.keras.models.load_model('song_speech_cnn_model.h5')
# Define feature extraction
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        mfccs = mfccs[np.newaxis, ..., np.newaxis]  # reshape for CNN
        return mfccs
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
# Define Gradio prediction function
def predict_song_speech(audio):
    features = extract_features(audio)
    if features is not None:
        prediction = model.predict(features)
        if prediction[0][0] < 0.5:
            return "🎵 This is a SONG."
        else:
            return "🗣️ This is a SPEECH."
    else:
        return "Error processing audio."
# Define Gradio interface
interface = gr.Interface(
    fn=predict_song_speech,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Song vs Speech Classifier",
    description="Upload an audio file to classify it as song or speech."
)
# Launch the app
if __name__ == "__main__":
    interface.launch()