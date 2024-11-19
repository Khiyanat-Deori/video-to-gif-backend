from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict, Any, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from slowapi import Limiter
from slowapi.util import get_remote_address
import ffmpeg
import whisper
import shutil
import uuid
import logging
import torch
import os
import asyncio
import firebase_admin
from firebase_admin import credentials, firestore
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Firebase
cred = credentials.Certificate('./serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_WORKERS = int(os.getenv('MAX_WORKERS', 3))
thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

DEVICE = "cpu"
torch.set_num_threads(1)
try:
    model = whisper.load_model("tiny", device=DEVICE)
    logger.info(f"Whisper model loaded successfully on {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

BASE_DIR = Path(os.getenv('RENDER_VOLUME_PATH', Path(__file__).parent.absolute()))
TEMP_DIR = BASE_DIR / "temp"
AUDIO_DIR = BASE_DIR / "audio"
FONT_DIR = BASE_DIR / "fonts"

for dir_path in [TEMP_DIR, AUDIO_DIR, FONT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {dir_path}")

def get_font() -> Path:
    """Get the Roboto font path from the fonts directory."""
    try:
        font_path = FONT_DIR / "Roboto-Regular.ttf"
        
        if not font_path.exists():
            raise FileNotFoundError(f"Required font not found at {font_path}. Please ensure Roboto-Regular.ttf is in the fonts directory.")
            
        logger.info(f"Using font: {font_path}")
        return font_path
        
    except Exception as e:
        logger.error(f"Font handling failed: {e}")
        raise

FONT_PATH = get_font()

async def save_to_firestore(gif_id: str, gif_path: Path, metadata: Dict) -> None:
    """Save GIF data and metadata to Firestore using chunks."""
    try:
        # Read the GIF file as bytes
        with open(gif_path, 'rb') as gif_file:
            gif_bytes = gif_file.read()
        
        # Convert to base64 string
        gif_data = base64.b64encode(gif_bytes).decode('utf-8')
        
        # Calculate number of chunks needed (max 900KB per chunk to be safe)
        chunk_size = 900000  # ~900KB in bytes
        total_chunks = (len(gif_data) + chunk_size - 1) // chunk_size
        
        # Create chunks collection for this GIF
        chunks_ref = db.collection('gifs').document(gif_id).collection('chunks')
        
        # Store metadata in the main document
        doc_data = {
            'created_at': firestore.SERVER_TIMESTAMP,
            'metadata': metadata,
            'total_chunks': total_chunks,
            'total_size': len(gif_data)
        }
        
        # Save main document
        await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: db.collection('gifs').document(gif_id).set(doc_data)
        )
        
        # Save chunks
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(gif_data))
            chunk_data = gif_data[start_idx:end_idx]
            
            await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: chunks_ref.document(str(i)).set({
                    'data': chunk_data,
                    'index': i
                })
            )
        
        logger.info(f"Successfully saved GIF {gif_id} to Firestore in {total_chunks} chunks")
    except Exception as e:
        logger.error(f"Failed to save to Firestore: {e}")
        # Clean up any chunks that were saved if there was an error
        try:
            await delete_from_firestore(gif_id)
        except Exception as cleanup_error:
            logger.error(f"Failed to cleanup after save error: {cleanup_error}")
        raise

async def delete_from_firestore(gif_id: str):
    """Delete a GIF document and its chunks from Firestore."""
    try:
        # Delete all chunks first
        chunks_ref = db.collection('gifs').document(gif_id).collection('chunks')
        chunks = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: chunks_ref.get()
        )
        
        for chunk in chunks:
            await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: chunk.reference.delete()
            )
        
        # Then delete the main document
        await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: db.collection('gifs').document(gif_id).delete()
        )
    except Exception as e:
        logger.error(f"Failed to delete from Firestore: {e}")
        raise

async def convert_video_to_audio(video_path: Path) -> Path:
    """Convert video to audio using ffmpeg-python."""
    try:
        output_path = AUDIO_DIR / f"{video_path.stem}_audio.wav"
        
        stream = (
            ffmpeg
            .input(str(video_path))
            .output(
                str(output_path),
                acodec='pcm_s16le',
                ar='16000',
                ac='1',
                loglevel='error'
            )
        )
        
        await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        )
        
        if not output_path.exists():
            raise Exception("Audio file was not created")
            
        return output_path
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise HTTPException(status_code=500, detail="Failed to convert video to audio")
    except Exception as e:
        logger.error(f"Failed to convert video to audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def transcribe_video(audio_file: str) -> dict:
    """Transcribe audio with specific Whisper settings for better segmentation."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: model.transcribe(
                audio_file,
                fp16=False,
                language='en',
                word_timestamps=True,
                condition_on_previous_text=False,
            )
        )
        logger.info(f"Transcription completed with {len(result['segments'])} segments")
        for seg in result['segments']:
            logger.info(f"Segment: {seg['text']} ({seg['start']:.2f}s - {seg['end']:.2f}s)")
            if 'words' in seg:
                for word in seg['words']:
                    logger.info(f"  Word: {word['word']} ({word['start']:.2f}-{word['end']:.2f})")
        return result
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

async def create_gif_with_synced_caption(
    video_path: Path,
    start_time: float,
    end_time: float,
    gif_id: str,
    words_with_timing: List[Dict]
) -> Dict[str, str]:
    """Create a GIF with synchronized captions and save to Firestore."""
    try:
        if not words_with_timing:
            return None
            
        temp_video = TEMP_DIR / f"{uuid.uuid4()}_captioned.mp4"
        palette_path = TEMP_DIR / f"{uuid.uuid4()}_palette.png"
        temp_gif = TEMP_DIR / f"{gif_id}.gif"
        font_path = str(FONT_PATH).replace('\\', '/').replace(':', r'\:')
        
        word_animations = []
        for word_timing in words_with_timing:
            escaped_word = word_timing["word"].upper().replace("'", "'\\''").replace(':', r'\:').replace('\\', r'\\')
            word_filter = (
                f"drawtext=fontfile='{font_path}'"
                f":text='{escaped_word}'"
                f":fontsize=50"
                f":fontcolor=white"
                f":bordercolor=yellow@1.0"
                f":borderw=3"
                f":box=1"
                f":boxcolor=black@0.5"
                f":boxborderw=5"
                f":x=(w-text_w)/2"
                f":y=h-text_h-20"
                f":enable='between(t,{word_timing['start']:.3f},{word_timing['end']:.3f})'"
            )
            word_animations.append(word_filter)
            
        complete_filter = ','.join(word_animations)
        duration = end_time - start_time

        try:
            stream = (
                ffmpeg
                .input(str(video_path), ss=start_time, t=duration)
                .output(
                    str(temp_video),
                    vf=complete_filter,
                    vcodec='libx264',
                    preset='ultrafast'
                )
                .overwrite_output()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            )

            stream = (
                ffmpeg
                .input(str(temp_video))
                .output(
                    str(palette_path),
                    vf='fps=15,scale=480:-1:flags=lanczos,palettegen=stats_mode=diff'
                )
                .overwrite_output()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            )

            stream = (
                ffmpeg
                .input(str(temp_video))
                .output(
                    str(temp_gif),
                    i=str(palette_path),
                    lavfi='fps=15,scale=480:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle'
                )
                .overwrite_output()
            )
            
            await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
            )

            # Save to Firestore
            metadata = {
                'duration': duration,
                'word_count': len(words_with_timing),
                'words': [word['word'] for word in words_with_timing]
            }
            
            await save_to_firestore(gif_id, temp_gif, metadata)
            
            return {
                "gif_id": gif_id
            }

        finally:
            # Cleanup temporary files
            for temp_file in [temp_video, palette_path, temp_gif]:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove temp file {temp_file}: {e}")

    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error in GIF creation: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error creating GIF: {str(e)}")
        return None
async def detect_silence(audio_path: Path, min_silence_duration: float = 0.3, silence_threshold: int = -35) -> List[Dict[str, float]]:
    """Detect silence using ffmpeg-python."""
    try:
        stream = (
            ffmpeg
            .input(str(audio_path))
            .filter('silencedetect', n=f"{silence_threshold}dB", d=str(min_silence_duration))
            .output('pipe:', format='null')
            .global_args('-loglevel', 'info')
        )
        
        _, stderr = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        )
        
        silence_periods = []
        silence_start = None
        
        for line in stderr.decode().split('\n'):
            if 'silence_start' in line:
                silence_start = float(line.split('silence_start: ')[1].split()[0])
            elif 'silence_end' in line and silence_start is not None:
                silence_end = float(line.split('silence_end: ')[1].split()[0])
                if silence_end - silence_start >= min_silence_duration:
                    silence_periods.append({
                        "start": silence_start,
                        "end": silence_end
                    })
                silence_start = None
        
        return silence_periods
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg silence detection error: {e.stderr.decode() if e.stderr else str(e)}")
        raise HTTPException(status_code=500, detail="Silence detection failed")
    except Exception as e:
        logger.error(f"Silence detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_video_duration(video_path: Path) -> float:
    """Get video duration using ffmpeg-python."""
    try:
        probe = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: ffmpeg.probe(str(video_path))
        )
        duration = float(probe['format']['duration'])
        return duration
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg probe error: {e.stderr.decode() if e.stderr else str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get video duration")
    except Exception as e:
        logger.error(f"Failed to get video duration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_speech_segments(silence_periods: List[Dict[str, float]], total_duration: float) -> List[Dict[str, float]]:
    """Convert silence periods into speech segments."""
    speech_segments = []
    current_time = 0.0
    
    for silence in silence_periods:
        if silence["start"] > current_time:
            speech_segments.append({
                "start": current_time,
                "end": silence["start"]
            })
        current_time = silence["end"]
    
    if current_time < total_duration:
        speech_segments.append({
            "start": current_time,
            "end": total_duration
        })
    
    for i, segment in enumerate(speech_segments):
        logger.info(f"Speech segment {i}: {segment['start']:.2f}-{segment['end']:.2f}")
    
    return speech_segments

async def process_video_segment_with_transcription(
    video_path: Path,
    segment: dict,
    transcription: dict,
    min_duration: float = 0.25,
    max_duration: float = 10.0
) -> Dict:
    """Process video segment and find corresponding transcription."""
    try:
        duration = segment["end"] - segment["start"]
        segment_start = segment["start"]
        segment_end = segment["end"]

        if duration < min_duration or duration > max_duration:
            logger.info(f"Skipping segment: duration {duration:.2f}s outside bounds ({min_duration}-{max_duration}s)")
            return None

        matching_words = []
        for trans_segment in transcription["segments"]:
            if "words" not in trans_segment:
                continue
                
            for word_data in trans_segment["words"]:
                word_start = word_data["start"]
                word_end = word_data["end"]

                overlap_start = max(segment_start, word_start)
                overlap_end = min(segment_end, word_end)
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > 0.1:
                    relative_start = max(0, overlap_start - segment_start)
                    relative_end = min(overlap_end - segment_start, duration)
                    
                    matching_words.append({
                        "word": word_data["word"],
                        "start": relative_start,
                        "end": relative_end,
                        "original_start": word_start,
                        "original_end": word_end
                    })
        
        matching_words.sort(key=lambda x: x["original_start"])
        
        if not matching_words:
            logger.info(f"No matching words found for segment {segment_start:.2f}-{segment_end:.2f}")
            return None
            
        caption_text = " ".join(word["word"] for word in matching_words)
        logger.info(f"Segment {segment_start:.2f}-{segment_end:.2f} words: {caption_text}")
        
        gif_id = str(uuid.uuid4())
        gif_result = await create_gif_with_synced_caption(
            video_path,
            segment_start,
            segment_end,
            gif_id,
            matching_words
        )
        
        if gif_result:
            return {
                "start": segment_start,
                "end": segment_end,
                "text": caption_text,
                "gif_id": gif_id,
                "duration": duration
            }
        return None
        
    except Exception as e:
        logger.error(f"Segment processing failed: {e}")
        return None

async def cleanup_files(files: List[Path]):
    """Clean up temporary files."""
    for file_path in files:
        try:
            if file_path and file_path.exists():
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    file_path.unlink
                )
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {str(e)}")

@app.post("/upload")
@limiter.limit("5/minute")
async def upload_video(
    request: Request,
    background_tasks: BackgroundTasks,
    min_silence_duration: float = 0.2,
    silence_threshold: int = -35,
    min_segment_duration: float = 0.25,
    max_segment_duration: float = 10.0,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Process uploaded video and create GIFs with synchronized captions."""
    temp_file = None
    audio_file = None
    
    try:
        temp_file = TEMP_DIR / f"{uuid.uuid4()}.mp4"
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        audio_file = await convert_video_to_audio(temp_file)
        
        total_duration = await get_video_duration(temp_file)
        
        transcription = await transcribe_video(str(audio_file))

        silence_periods = await detect_silence(
            audio_file,
            min_silence_duration=min_silence_duration,
            silence_threshold=silence_threshold
        )
        
        speech_segments = get_speech_segments(silence_periods, total_duration)
        
        tasks = [
            process_video_segment_with_transcription(
                temp_file, 
                segment, 
                transcription,
                min_segment_duration,
                max_segment_duration
            )
            for segment in speech_segments
        ]
        
        segments = await asyncio.gather(*tasks)
        
        successful_segments = [s for s in segments if s is not None]
        failed_segments = [
            {"start": orig["start"], "end": orig["end"]}
            for s, orig in zip(segments, speech_segments)
            if s is None
        ]
        
        background_tasks.add_task(cleanup_files, [temp_file, audio_file])
        
        return {
            "message": "Processing complete",
            "successful_segments": successful_segments,
            "failed_segments": failed_segments,
            "total_segments": len(speech_segments),
            "successful_count": len(successful_segments),
            "failed_count": len(failed_segments)
        }
        
    except Exception as e:
        if temp_file or audio_file:
            background_tasks.add_task(cleanup_files, [temp_file, audio_file])
        logger.error(f"Upload processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gif/{gif_id}")
async def get_gif(gif_id: str):
    """Retrieve a generated GIF by its ID from Firestore."""
    try:
        # Get the main document
        doc_ref = db.collection('gifs').document(gif_id)
        doc = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: doc_ref.get()
        )
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="GIF not found")
        
        doc_data = doc.to_dict()
        total_chunks = doc_data['total_chunks']
        
        # Get all chunks
        chunks_ref = doc_ref.collection('chunks')
        chunks = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: chunks_ref.get()
        )
        
        # Sort chunks by index and concatenate data
        sorted_chunks = sorted(chunks, key=lambda x: x.to_dict()['index'])
        gif_data_b64 = ''.join(chunk.to_dict()['data'] for chunk in sorted_chunks)
        
        # Decode base64 data
        gif_data = base64.b64decode(gif_data_b64)
        
        return Response(
            content=gif_data,
            media_type="image/gif",
            headers={"Content-Disposition": f"inline; filename={gif_id}.gif"}
        )
    except Exception as e:
        logger.error(f"Failed to retrieve GIF: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve GIF")
    
@app.get("/health")
async def health_check():
    """Check the health status of the service."""
    return {
        "status": "healthy",
        "device": DEVICE,
        "font_path": str(FONT_PATH),
        "max_workers": MAX_WORKERS,
        "firebase_initialized": firebase_admin._apps is not None,
        "directories": {
            "temp": str(TEMP_DIR),
            "audio": str(AUDIO_DIR)
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


