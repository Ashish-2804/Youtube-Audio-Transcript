#  AI Agent for YouTube Video Transcript

This project builds an AI agent that automatically fetches, transcribes, and interacts with YouTube video content. It supports searching through transcripts, summarizing the video, and locating timestamps for specific topics.

##  Features

-  YouTube audio download
-  Audio-to-text transcription using OpenAI Whisper
-  Keyword/topic search within transcript
-  Extract timestamps for keywords
-  Summarization
-  Streamlit UI

##  Technologies

- Python
- pytube
- OpenAI Whisper
- re, pandas
- NLTK - (Summarization)
- Streamlit 

Use ffmpeg for multimedia files particularly for audio and video
- choco install ffmpeg

## 📦 Installation

```bash
Clone the repo: Using git clone 

choco install ffmpeg
pip install -r requirements.txt
streamlit run main.py
