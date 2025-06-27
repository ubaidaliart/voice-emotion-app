import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
from transformers import pipeline
from pydub import AudioSegment
import torch

# Initialize emotion detector
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "audio-classification", 
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

# Process audio to standard format
def process_audio(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio = AudioSegment.from_file(uploaded_file)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(tmp.name, format="wav")
        return tmp.name

# Main app
def main():
    st.set_page_config(page_title="Voice Emotion Detector", layout="wide")
    st.title("🎤 Real-Time Emotion Detection")
    st.markdown("Upload customer call recordings to detect emotions using AI")
    
    # Sidebar for upload
    with st.sidebar:
        st.header("Upload Call Recording")
        uploaded_file = st.file_uploader("Choose audio file", 
                                        type=["wav", "mp3", "m4a", "ogg", "flac"],
                                        accept_multiple_files=False)
        
        if uploaded_file:
            st.audio(uploaded_file)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    with col1:
        st.subheader("Analysis Controls")
        
        if uploaded_file:
            if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    processed_path = process_audio(uploaded_file)
                    emotion_classifier = load_emotion_model()
                    
                    try:
                        preds = emotion_classifier(processed_path)
                        result = {
                            "file": uploaded_file.name,
                            "emotion": preds[0]['label'],
                            "confidence": f"{preds[0]['score']:.0%}",
                            "score": preds[0]['score']
                        }
                        st.session_state.results.append(result)
                        st.success("Analysis complete!")
                        os.unlink(processed_path)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        if st.session_state.results:
            latest = st.session_state.results[-1]
            
            # Emotion display
            emotion_color = {
                "angry": "#FF4B4B",
                "happy": "#00D26A",
                "sad": "#1C83E1",
                "neutral": "#7E7E7E"
            }.get(latest['emotion'].lower(), "#7E7E7E")
            
            st.subheader("Current Results")
            st.markdown(f"""
            <div style="background:{emotion_color};padding:20px;border-radius:10px">
                <h2 style="color:white;text-align:center;">{latest['emotion'].upper()}</h2>
                <h3 style="color:white;text-align:center;">Confidence: {latest['confidence']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Emotion distribution
            st.subheader("Emotion Distribution")
            if len(st.session_state.results) > 1:
                df = pd.DataFrame(st.session_state.results)
                fig = px.pie(df, names='emotion', hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Upload more files to see distribution")
    
    # History table
    if st.session_state.results:
        st.subheader("Analysis History")
        history_df = pd.DataFrame(st.session_state.results)
        st.dataframe(
            history_df[['file', 'emotion', 'confidence']].sort_index(ascending=False),
            hide_index=True,
            use_container_width=True
        )
        
        # Export button
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Export Results as CSV",
            csv,
            "emotion_results.csv",
            "text/csv",
            key='download-csv'
        )

if __name__ == "__main__":
    main()
