# YTTranscribe

YTTranscribe is a powerful Streamlit application that allows you to generate transcripts from YouTube videos using OpenAI's Whisper speech recognition model. The application provides a user-friendly interface to:

- Download audio from YouTube videos
- Transcribe the audio using various Whisper model sizes
- View the full transcript with timestamps
- Search for specific words or phrases in the transcript
- Generate summaries of the video content

## Features

- **Multiple Whisper Model Options**: Choose from tiny, base, small, medium, or large models based on your accuracy needs and system capabilities
- **Interactive YouTube Video Display**: Watch the video while reviewing its transcript
- **Timestamp Navigation**: Click on timestamps to jump to specific points in the video
- **Text Search**: Find all occurrences of specific words or phrases with timestamp links
- **Summarization**: Generate concise summaries of video content with adjustable length
- **Export Options**: Download transcripts and summaries as text files

## Installation

### Prerequisites

- Python 3.7 or higher
- FFmpeg (required for audio extraction)

### Installing FFmpeg

#### Windows:
1. Download the FFmpeg build from the [official FFmpeg website](https://ffmpeg.org/download.html#build-windows) or use the [gyan.dev build](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the downloaded ZIP file to a location on your computer (e.g., `C:\ffmpeg`)
3. Add FFmpeg to your system PATH:
   - Open the Start menu and search for "Environment Variables"
   - Click on "Edit the system environment variables"
   - Click the "Environment Variables" button
   - Under "System variables", find and select the "Path" variable, then click "Edit"
   - Click "New" and add the path to the FFmpeg `bin` folder (e.g., `C:\ffmpeg\bin`)
   - Click "OK" on all dialogs to save changes
4. Verify the installation by opening a new Command Prompt and typing:
   ```
   ffmpeg -version
   ```

#### Linux:
- **Ubuntu/Debian**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **Fedora**:
  ```bash
  sudo dnf install ffmpeg
  ```

- **Arch Linux**:
  ```bash
  sudo pacman -S ffmpeg
  ```

- Verify the installation:
  ```bash
  ffmpeg -version
  ```

### Setting up YTTranscribe

1. Clone this repository or download the source code

2. Navigate to the project directory:
   ```bash
   cd YTTranscribe
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```

2. The application will open in your default web browser

3. In the sidebar:
   - Select a Whisper model size (larger models are more accurate but slower)
   - Enter a YouTube URL 
   - Click "Generate Transcript"

4. Once processing is complete, you can:
   - View the full transcript with timestamps
   - Search for specific words or phrases
   - Generate a summary of the content

## System Requirements

The resource requirements depend on the Whisper model size you select:

- **Tiny/Base**: Works on most systems with 4GB+ RAM
- **Small**: Recommended 8GB+ RAM
- **Medium**: Recommended 16GB+ RAM
- **Large**: Recommended 16GB+ RAM with a powerful CPU or GPU

Processing time will vary based on your system specifications and the length of the video.

## Troubleshooting

- **Download Errors**: Some videos may be protected, age-restricted, or region-blocked
- **Processing Time**: Longer videos will take more time to process, especially with larger models
- **Memory Issues**: If you encounter memory errors, try using a smaller Whisper model

## License

[MIT License](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [Streamlit](https://streamlit.io/) for the web interface
