import os
import re
from datetime import timedelta
import streamlit as st
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
import time
import whisper
import yt_dlp as youtube_dl

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class YouTubeTranscriptAgent:
    def __init__(self, model_size="base"):
        """
        Initialize the YouTube Transcript Agent.
        
        Args:
            model_size (str): Size of the Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model = whisper.load_model(model_size)
        self.transcript_df = None
        self.video_info = {}
        
    def download_audio(self, youtube_url, output_dir=None):
        """
        Download audio from a YouTube video using yt-dlp.
        
        Args:
            youtube_url (str): URL of the YouTube video
            output_dir (str): Directory to save the downloaded audio
            
        Returns:
            str: Path to the downloaded audio file
        """
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "audio")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract video ID from the URL
        video_id = None
        if "youtube.com/watch" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = "video"  # Fallback name
            
        output_file = os.path.join(output_dir, f"{video_id}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_file,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }
        
        # Download video info first
        info_opts = {
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with youtube_dl.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Store video information
                self.video_info = {
                    'title': info.get('title', 'Unknown'),
                    'author': info.get('uploader', 'Unknown'),
                    'publish_date': info.get('upload_date', 'Unknown'),
                    'views': info.get('view_count', 0),
                    'length': info.get('duration', 0),
                    'url': youtube_url,
                    'thumbnail_url': info.get('thumbnail', '')
                }
            
            # Now download the audio
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                
            # The actual filename might be different because of yt-dlp's processing
            output_file = os.path.join(output_dir, f"{video_id}.mp3")
            
            return output_file
            
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file using Whisper.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Transcription result
        """
        result = self.model.transcribe(audio_path, fp16=False)
        
        # Create DataFrame from segments
        data = []
        for segment in result['segments']:
            data.append({
                'start': segment['start'],
                'end': segment['end'],
                'start_str': str(timedelta(seconds=int(segment['start']))),
                'text': segment['text'].strip()
            })
        
        self.transcript_df = pd.DataFrame(data)
        return result
    
    def search_transcript(self, query):
        """
        Search for a query in the transcript.
        
        Args:
            query (str): Text to search for
            
        Returns:
            DataFrame: Matching segments
        """
        if self.transcript_df is None:
            st.error("No transcript available. Please transcribe a video first.")
            return None
        
        pattern = re.compile(query, re.IGNORECASE)
        matches = self.transcript_df[self.transcript_df['text'].str.contains(pattern)]
        
        return matches
    
    def summarize_transcript(self, num_sentences=5):
        """
        Generate a simple extractive summary of the transcript.
        
        Args:
            num_sentences (int): Number of sentences to include in summary
            
        Returns:
            str: Summary text
        """
        if self.transcript_df is None:
            st.error("No transcript available. Please transcribe a video first.")
            return None
        
        # Combine all transcript text
        full_text = " ".join(self.transcript_df['text'].tolist())
        
        # Split into sentences
        sentences = sent_tokenize(full_text)
        
        # For this simple implementation, just take the first few sentences
        # A more sophisticated approach would use TextRank or another algorithm
        if num_sentences >= len(sentences):
            summary = " ".join(sentences)
        else:
            summary = " ".join(sentences[:num_sentences])
        
        return summary
    
    def get_timestamps(self, query):
        """
        Get timestamps for segments containing the query.
        
        Args:
            query (str): Text to search for
            
        Returns:
            list: List of (timestamp, text) tuples
        """
        matches = self.search_transcript(query)
        if matches is None or len(matches) == 0:
            return []
        
        timestamps = [(row['start_str'], row['text']) for _, row in matches.iterrows()]
        return timestamps
    
    def get_full_transcript(self):
        """
        Get the full transcript as a formatted string.
        
        Returns:
            str: Formatted transcript
        """
        if self.transcript_df is None:
            return "No transcript available."
        
        transcript_text = ""
        for _, row in self.transcript_df.iterrows():
            transcript_text += f"[{row['start_str']}] {row['text']}\n"
        
        return transcript_text

def format_time(seconds):
    """Format seconds to mm:ss or hh:mm:ss."""
    td = timedelta(seconds=seconds)
    if td.seconds < 3600:
        return f"{td.seconds // 60:02d}:{td.seconds % 60:02d}"
    else:
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_youtube_embed_url(youtube_url):
    """Convert a YouTube video URL to an embedded URL."""
    video_id = None
    
    # Extract video ID from URL
    if "youtube.com/watch" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    
    if video_id:
        return f"https://www.youtube.com/embed/{video_id}"
    return None

def main():
    st.set_page_config(
        page_title="YouTube Transcript AI Agent",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("🎬 YouTube Transcript AI Agent")
    
    # Initialize session state variables if they don't exist
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'transcript_generated' not in st.session_state:
        st.session_state.transcript_generated = False
    if 'video_info' not in st.session_state:
        st.session_state.video_info = {}
    
    # Sidebar for model selection and URL input
    with st.sidebar:
        st.header("Settings")
        model_size = st.selectbox(
            "Select Whisper Model Size",
            options=["tiny", "base", "small", "medium", "large"],
            index=1,  # Default to 'base'
            help="Larger models are more accurate but slower and require more memory"
        )
        
        youtube_url = st.text_input("YouTube Video URL", "")
        
        if st.button("Generate Transcript"):
            if youtube_url:
                try:
                    with st.spinner("Initializing Whisper model..."):
                        st.session_state.agent = YouTubeTranscriptAgent(model_size=model_size)
                    
                    with st.spinner("Downloading audio..."):
                        audio_path = st.session_state.agent.download_audio(youtube_url)
                    
                    with st.spinner("Transcribing audio... This may take a few minutes."):
                        st.session_state.agent.transcribe_audio(audio_path)
                    
                    st.session_state.transcript_generated = True
                    st.session_state.video_info = st.session_state.agent.video_info
                    st.success("Transcript generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    st.info("If you're experiencing download issues, make sure the video is publicly accessible and not age-restricted.")
            else:
                st.warning("Please enter a YouTube URL")
    
    # Main content area - display transcript and features when available
    if st.session_state.transcript_generated and st.session_state.agent:
        # Create two columns for video info and video player
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Video Information")
            st.write(f"**Title:** {st.session_state.video_info.get('title', 'Unknown')}")
            st.write(f"**Channel:** {st.session_state.video_info.get('author', 'Unknown')}")
            st.write(f"**Duration:** {format_time(st.session_state.video_info.get('length', 0))}")
            st.write(f"**Views:** {st.session_state.video_info.get('views', 'Unknown')}")
        
        with col2:
            # Display embedded YouTube video
            embed_url = create_youtube_embed_url(st.session_state.video_info.get('url', ''))
            if embed_url:
                st.components.v1.iframe(embed_url, height=300, width=None)
            else:
                st.image(st.session_state.video_info.get('thumbnail_url', ''))
        
        # Tabs for different features
        tab1, tab2, tab3 = st.tabs(["Full Transcript", "Search & Timestamps", "Summarize"])
        
        with tab1:
            st.subheader("Full Transcript")
            transcript_text = st.session_state.agent.get_full_transcript()
            st.text_area("", transcript_text, height=400)
            
            # Download button for transcript
            st.download_button(
                label="Download Transcript",
                data=transcript_text,
                file_name=f"{st.session_state.video_info.get('title', 'transcript')}.txt",
                mime="text/plain"
            )
        
        with tab2:
            st.subheader("Search Transcript")
            search_query = st.text_input("Enter search term:", "")
            
            if search_query:
                matches = st.session_state.agent.search_transcript(search_query)
                
                if matches is not None and len(matches) > 0:
                    st.success(f"Found {len(matches)} matches for '{search_query}'")
                    
                    # Create a DataFrame for display
                    display_df = matches[['start_str', 'text']].copy()
                    display_df.columns = ['Timestamp', 'Text']
                    
                    # Add YouTube time links
                    video_id = None
                    if "youtube.com/watch" in st.session_state.video_info.get('url', ''):
                        video_id = st.session_state.video_info.get('url', '').split("v=")[1].split("&")[0]
                    elif "youtu.be/" in st.session_state.video_info.get('url', ''):
                        video_id = st.session_state.video_info.get('url', '').split("youtu.be/")[1].split("?")[0]
                    
                    if video_id:
                        display_df['Link'] = display_df.apply(
                            lambda row: f"[Go to timestamp](https://www.youtube.com/watch?v={video_id}&t={int(matches.loc[row.name, 'start'])}s)", 
                            axis=1
                        )
                    
                    st.dataframe(display_df, use_container_width=True)
                else:
                    st.info(f"No matches found for '{search_query}'")
        
        with tab3:
            st.subheader("Summarize Transcript")
            num_sentences = st.slider("Number of sentences in summary:", 3, 20, 5)
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = st.session_state.agent.summarize_transcript(num_sentences)
                    st.markdown("### Summary")
                    st.write(summary)
                    
                    # Download button for summary
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"{st.session_state.video_info.get('title', 'summary')}_summary.txt",
                        mime="text/plain"
                    )
    else:
        # Display instructions when no transcript is available
        st.info("👈 Enter a YouTube URL in the sidebar and click 'Generate Transcript' to begin.")
        
        st.markdown("""
        ### How to use this tool:
        
        1. Select a Whisper model size in the sidebar (larger models are more accurate but slower)
        2. Enter a YouTube URL in the sidebar
        3. Click 'Generate Transcript' to process the video
        4. Once processing is complete, you can:
           - View the full transcript
           - Search for specific words or phrases
           - Generate a summary of the content
        
        ### Requirements:
        - Internet connection to download the YouTube video and Whisper model
        - Sufficient memory for the selected model size
        
        ### Troubleshooting:
        - If you encounter download errors, try a different video
        - Some videos may be protected or region-restricted
        - Larger videos may require more processing time
        """)

if __name__ == "__main__":
    main()