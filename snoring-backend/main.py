import os
import io
import tempfile
import base64

from database import SessionLocal, Base, engine
from db_models.session import SnoringSession

from datetime import datetime  # Added missing import
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import detect_silence
import logging
from typing import Optional, Dict, Any, Tuple


Base.metadata.create_all(bind=engine)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib to use Agg backend for non-GUI environments
import matplotlib
matplotlib.use('Agg')

app = FastAPI(title="Snoring Detection API", version="1.0.0")

# CORS middleware to allow requests from your Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
MODEL_PATHS = [
    'models/best_snoring_model.keras',
    'models/final_snoring_detection_model.h5',
    'best_snoring_model.keras',
    'final_snoring_detection_model.h5',
    'snoring_model.keras',  # Added as fallback
    'snoring_model.h5'      # Added as fallback
]

# Audio processing parameters
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = -40  # dBFS threshold for silence detection
MIN_SILENCE_LEN = 500   # Minimum silence length in ms
SILENCE_PADDING = 100   # Padding around silence in ms

def create_model_architecture():
    """Create the exact model architecture used in training"""
    def create_improved_model():
        """Create the improved CNN model matching your architecture"""
        model = tf.keras.Sequential([
            # Input layer with proper shape
            tf.keras.layers.InputLayer(input_shape=(61, 257, 1)),
            
            # First conv block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third conv block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth conv block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Global pooling and dense layers
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),  # Fixed typo: 'reelu' -> 'relu'
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    return create_improved_model()

def load_model_with_architecture(model_path: str):
    """Load model by first creating architecture then loading weights"""
    try:
        # Method 1: Try direct load first (for newer TensorFlow formats)
        logger.info(f"Attempting direct model load from: {model_path}")
        
        # Try different loading methods
        try:
            # Try loading with custom objects
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=None,
                compile=False
            )
            logger.info("✅ Model loaded successfully with direct method")
            return model
        except Exception as e1:
            logger.warning(f"Direct load attempt 1 failed: {e1}")
            
            # Try another method
            try:
                model = tf.keras.models.load_model(model_path)
                logger.info("✅ Model loaded successfully with alternative method")
                return model
            except Exception as e2:
                logger.warning(f"Direct load attempt 2 failed: {e2}")
        
        # Method 2: Create architecture and load weights
        logger.info("Creating model architecture and loading weights...")
        model = create_model_architecture()
        
        # Try loading weights with different approaches
        try:
            # Try loading weights directly
            model.load_weights(model_path)
            logger.info("✅ Model weights loaded successfully with direct weights")
            
        except Exception as weights_error:
            logger.warning(f"Direct weights load failed: {weights_error}")
            
            # Try with by_name if it's a full model save
            try:
                # Load the model file to extract weights
                temp_model = tf.keras.models.load_model(model_path, compile=False)
                # Get weights from temp model and set them in our model
                for layer in model.layers:
                    try:
                        # Try to find matching layer in temp model
                        for temp_layer in temp_model.layers:
                            if layer.name == temp_layer.name:
                                layer.set_weights(temp_layer.get_weights())
                                break
                    except:
                        continue
                logger.info("✅ Model weights transferred successfully")
            except Exception as transfer_error:
                logger.error(f"Weight transfer failed: {transfer_error}")
                return None
        
        # Compile the model (not necessary for inference but good practice)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
                
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

@app.on_event("startup")
async def load_model():
    """Load the TensorFlow model on startup"""
    global model
    try:
        logger.info("=" * 50)
        logger.info("Starting model loading process...")
        logger.info("=" * 50)
        
        for model_path in MODEL_PATHS:
            if os.path.exists(model_path):
                logger.info(f"\n🔍 Found model at: {model_path}")
                logger.info(f"📊 File size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
                
                model = load_model_with_architecture(model_path)
                if model is not None:
                    logger.info(f"✅ Model successfully loaded from: {model_path}")
                    
                    # Test the model with a dummy input to verify it works
                    try:
                        logger.info("🧪 Testing model with dummy input...")
                        dummy_input = np.random.random((1, 61, 257, 1)).astype(np.float32)
                        prediction = model.predict(dummy_input, verbose=0)
                        logger.info(f"✅ Model test prediction: {prediction[0][0]:.4f}")
                        
                        # Print model summary
                        logger.info("📋 Model Summary:")
                        model.summary(print_fn=lambda x: logger.info(f"  {x}"))
                        
                    except Exception as test_error:
                        logger.warning(f"⚠️ Model test failed: {test_error}")
                    
                    break
                else:
                    logger.warning(f"❌ Failed to load model from: {model_path}")
        else:
            logger.error("\n❌ No model file found or all loading attempts failed.")
            logger.info("\n📁 Please ensure model files are in one of these locations:")
            for path in MODEL_PATHS:
                exists = "✅ EXISTS" if os.path.exists(path) else "❌ NOT FOUND"
                logger.info(f"  - {path} {exists}")
            
            # Try to list current directory
            logger.info("\n📂 Current directory contents:")
            try:
                for item in os.listdir('.'):
                    if item.endswith(('.keras', '.h5', '.hdf5')):
                        logger.info(f"  - {item} (potential model file)")
                    elif os.path.isdir(item):
                        logger.info(f"  - {item}/ (directory)")
            except Exception as e:
                logger.error(f"Error listing directory: {e}")
                
    except Exception as e:
        logger.error(f"❌ Error during model loading process: {e}")
        model = None
    
    # If model still not loaded, try a simple model
    if model is None:
        logger.warning("\n⚠️ Could not load pre-trained model. Creating a simple model for testing...")
        try:
            model = create_model_architecture()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ Created simple model for testing purposes")
        except Exception as e:
            logger.error(f"❌ Failed to create simple model: {e}")

@app.get("/")
async def root():
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "message": "Snoring Detection API", 
        "status": "running",
        "model_loaded": model is not None,
        "model_status": model_status
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for connection testing"""
    model_status = "loaded" if model is not None else "not loaded"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_status": model_status,
        "timestamp": str(datetime.now())  # Now datetime is imported
    }

# 🌙 Night Preferences (PATCH – minimal, in-memory)
NIGHT_PREFERENCES: Dict[str, Any] = {}

@app.post("/night-preferences")
async def save_night_preferences(prefs: Dict[str, Any]):
    """
    Save night preferences sent from frontend.
    Non-persistent, in-memory storage (sufficient for project scope).
    """
    NIGHT_PREFERENCES.update(prefs)
    logger.info(f"🌙 Night preferences updated: {prefs}")
    return {"status": "saved"}

from datetime import time

def is_within_quiet_hours(now: time, start: time, end: time) -> bool:
    """
    Handles quiet hours that may span across midnight (e.g. 23:00 → 07:00)
    """
    if start <= end:
        return start <= now <= end
    else:
        return now >= start or now <= end



def remove_silence_from_audio(audio: AudioSegment) -> AudioSegment:
    """Remove silent portions from audio"""
    try:
        # Detect silence in the audio
        silence_ranges = detect_silence(
            audio, 
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESHOLD,
            seek_step=1
        )
        
        if not silence_ranges:
            logger.info("No significant silence detected in audio")
            return audio
        
        logger.info(f"Detected {len(silence_ranges)} silent regions")
        
        # Create a new audio segment without silence
        non_silent_audio = AudioSegment.empty()
        last_end = 0
        
        for start, end in silence_ranges:
            # Add padding
            start = max(0, start - SILENCE_PADDING)
            end = min(len(audio), end + SILENCE_PADDING)
            
            # Add non-silent part
            if start > last_end:
                non_silent_audio += audio[last_end:start]
            
            last_end = end
        
        # Add remaining audio after last silence
        if last_end < len(audio):
            non_silent_audio += audio[last_end:]
        
        logger.info(f"Audio length reduced from {len(audio)/1000:.1f}s to {len(non_silent_audio)/1000:.1f}s after removing silence")
        
        # Check if audio is too short after removing silence
        if len(non_silent_audio) < 1000:  # Less than 1 second
            logger.warning("Audio is very short after removing silence, using original audio")
            return audio
            
        return non_silent_audio
        
    except Exception as e:
        logger.error(f"Error removing silence: {e}")
        return audio

def convert_to_mp3(file_content: bytes, original_filename: str) -> Tuple[bytes, float]:
    """
    Convert any audio file to MP3 format and remove silence.
    
    Args:
        file_content: Raw bytes of the audio file
        original_filename: Original filename to determine format
        
    Returns:
        Tuple of (MP3 file content as bytes, original_duration_in_seconds)
    """
    try:
        # Determine file format from extension
        file_extension = original_filename.lower().split('.')[-1] if '.' in original_filename else ''
        
        # Create a temporary file with the original extension
        with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as temp_input:
            temp_input.write(file_content)
            temp_input_path = temp_input.name
        
        try:
            # Load the audio file using pydub (it auto-detects format)
            logger.info(f"Loading audio file with extension: {file_extension}")
            
            # Try to load with specific format first, then auto-detect
            try:
                audio = AudioSegment.from_file(temp_input_path, format=file_extension if file_extension in ['mp3', 'wav', 'm4a', 'mp4'] else None)
            except:
                audio = AudioSegment.from_file(temp_input_path)
            
            original_duration = len(audio) / 1000.0  # Convert to seconds
            logger.info(f"Original audio duration: {original_duration:.2f} seconds")
            
            # Check if audio has sound
            if audio.dBFS == float('-inf') or audio.max_dBFS == float('-inf'):
                logger.warning("Audio appears to be completely silent or has no sound")
                raise HTTPException(status_code=400, detail="The uploaded audio file appears to be completely silent. Please record with sound.")
            
            # Remove silence from audio
            audio = remove_silence_from_audio(audio)
            
            # Normalize audio volume
            audio = audio.normalize()
            
            # Set consistent format
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
            
            # Check audio energy
            rms = audio.rms
            if rms < 100:  # Very low RMS value
                logger.warning(f"Audio has very low RMS value: {rms}")
            
            # Export to MP3 in memory
            mp3_buffer = io.BytesIO()
            audio.export(mp3_buffer, format="mp3", parameters=["-q:a", "2"])  # Quality setting 2 (good quality)
            mp3_buffer.seek(0)
            mp3_content = mp3_buffer.read()
            
            logger.info(f"✅ Successfully converted {original_filename} to MP3 format")
            logger.info(f"Original size: {len(file_content)} bytes, MP3 size: {len(mp3_content)} bytes")
            logger.info(f"Processed audio duration: {len(audio)/1000:.2f} seconds")
            
            return mp3_content, original_duration
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting {original_filename} to MP3: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")

def convert_mp3_to_wav(mp3_file_content: bytes) -> io.BytesIO:
    """Convert MP3 file content to WAV format in memory"""
    try:
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
            temp_mp3.write(mp3_file_content)
            temp_mp3_path = temp_mp3.name
        
        # Load and convert MP3
        audio = AudioSegment.from_file(temp_mp3_path, format="mp3")
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        
        # Export to bytes buffer
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Clean up temporary file
        os.unlink(temp_mp3_path)
        
        return wav_buffer
        
    except Exception as e:
        logger.error(f"Error converting MP3 to WAV: {e}")
        raise HTTPException(status_code=400, detail=f"MP3 conversion failed: {str(e)}")

def load_mp3_as_tensor(mp3_file_content: bytes):
    """Load MP3 file content and convert to TensorFlow tensor"""
    try:
        wav_buffer = convert_mp3_to_wav(mp3_file_content)
        audio_binary = wav_buffer.read()
        
        # Decode WAV using TensorFlow
        wav, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        
        logger.info(f"MP3 loaded successfully. Length: {len(wav)} samples, Duration: {len(wav)/SAMPLE_RATE:.2f} seconds")
        
        # Check if audio has signal
        wav_np = wav.numpy()
        if np.max(np.abs(wav_np)) < 0.01:  # Very low amplitude
            logger.warning("Audio tensor has very low amplitude, may be silent")
        
        return wav
        
    except Exception as e:
        logger.error(f"Error loading MP3 as tensor: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")

def create_spectrogram(segment):
    """Create spectrogram from audio segment matching your preprocessing"""
    frame_length = 512
    frame_step = 256
    
    spectrogram = tf.signal.stft(segment, frame_length=frame_length, frame_step=frame_step)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    # Normalize
    mean = tf.math.reduce_mean(spectrogram)
    std = tf.math.reduce_std(spectrogram)
    spectrogram = (spectrogram - mean) / (std + 1e-6)
    
    # Expand dimensions for CNN input
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # Add channel dimension
    return spectrogram

def create_visualizations(wav, predictions, snoring_segments, filename, audio_duration):
    """Create comprehensive visualizations and return as base64 strings"""
    try:
        visualizations = {}
        
        # Create a larger figure for better mobile display
        plt.style.use('default')
        
        # Figure 1: Comprehensive Analysis Plot (similar to your original)
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle(f'Snoring Analysis: {filename}', fontsize=16, fontweight='bold')
        
        # Plot 1: Audio waveform
        time_axis = np.linspace(0, len(wav)/SAMPLE_RATE, len(wav))
        ax1.plot(time_axis, wav.numpy(), alpha=0.7, color='blue', linewidth=0.5)
        ax1.set_title('Audio Waveform')
        ax1.set_ylabel('Amplitude')
        ax1.set_xlabel('Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Snoring probability over time
        segment_times = [(i * 0.5) for i in range(len(predictions))]
        ax2.plot(segment_times, predictions, 'o-', color='red', alpha=0.7, markersize=3, linewidth=1)
        ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
        ax2.set_title('Snoring Probability Over Time')
        ax2.set_ylabel('Snoring Probability')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Snoring segments visualization
        for i, is_snoring in enumerate(snoring_segments):
            color = 'red' if is_snoring else 'green'
            alpha = 0.6 if is_snoring else 0.3
            ax3.axvspan(i * 0.5, (i + 1) * 0.5, color=color, alpha=alpha)
        ax3.set_title('Snoring Detection Timeline')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Snoring')
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['No', 'Yes'])
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Snoring distribution pie chart
        snoring_count = sum(snoring_segments)
        non_snoring_count = len(snoring_segments) - snoring_count
        labels = ['Snoring', 'Non-Snoring']
        sizes = [snoring_count, non_snoring_count]
        colors = ['#ff6b6b', '#51cf66']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Snoring Distribution')
        
        plt.tight_layout()
        
        # Convert to base64
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0)
        visualizations['analysis_plot'] = base64.b64encode(buf1.read()).decode('utf-8')
        plt.close(fig1)
        
        # Figure 2: Statistics and Metrics
        fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Confidence distribution
        ax1.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Prediction Confidence Distribution')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Snoring intervals duration
        interval_durations = []
        current_duration = 0
        for is_snoring in snoring_segments:
            if is_snoring:
                current_duration += 0.5
            else:
                if current_duration > 0:
                    interval_durations.append(current_duration)
                    current_duration = 0
        if current_duration > 0:
            interval_durations.append(current_duration)
            
        if interval_durations:
            ax2.hist(interval_durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax2.set_title('Snoring Interval Durations')
            ax2.set_xlabel('Duration (seconds)')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'No Snoring Intervals', ha='center', va='center', transform=ax2.transAxes)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Snoring pattern over time (smoothed)
        from scipy.ndimage import gaussian_filter1d
        if len(predictions) > 10:
            smoothed_predictions = gaussian_filter1d(predictions, sigma=2)
            ax3.plot(segment_times, smoothed_predictions, color='purple', linewidth=2)
            ax3.set_title('Smoothed Snoring Pattern')
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Snoring Probability')
            ax3.set_ylim(0, 1)
        else:
            ax3.text(0.5, 0.5, 'Insufficient Data\nfor Smoothing', ha='center', va='center', transform=ax3.transAxes)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        stats_text = f"""
        Analysis Summary:
        
        Total Duration: {audio_duration:.1f}s
        Segments Analyzed: {len(predictions)}
        Snoring Segments: {snoring_count}
        Snoring Ratio: {snoring_count/len(predictions)*100:.1f}%
        Snoring Intervals: {len(interval_durations)}
        Avg Interval Duration: {np.mean(interval_durations):.1f}s
        Max Interval Duration: {np.max(interval_durations) if interval_durations else 0:.1f}s
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Analysis Statistics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Convert to base64
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        visualizations['statistics_plot'] = base64.b64encode(buf2.read()).decode('utf-8')
        plt.close(fig2)
        
        # Figure 3: Simple timeline for mobile (vertical layout)
        fig3, ax = plt.subplots(figsize=(12, 6))
        
        # Create a simple timeline
        ax.plot(segment_times, predictions, 'o-', color='red', alpha=0.7, markersize=2, linewidth=1)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold')
        
        # Highlight snoring regions
        for i, is_snoring in enumerate(snoring_segments):
            if is_snoring:
                ax.axvspan(i * 0.5, (i + 1) * 0.5, color='red', alpha=0.3)
        
        ax.set_title('Snoring Detection Timeline')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Snoring Probability')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        buf3.seek(0)
        visualizations['timeline_plot'] = base64.b64encode(buf3.read()).decode('utf-8')
        plt.close(fig3)
        
        logger.info("✅ All visualizations generated successfully")
        return visualizations
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return {}

def preprocess_mp3_for_prediction(mp3_file_content: bytes, filename: str):
    """Preprocess MP3 file and make predictions"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Load MP3 as tensor
    wav = load_mp3_as_tensor(mp3_file_content)
    
    if wav is None:
        raise HTTPException(status_code=400, detail="Failed to load MP3 file")
    
    # Process the entire audio file in segments
    segment_length = 16000  # 1 second segments
    hop_length = 8000       # 0.5 second hop (50% overlap)
    
    segments = []
    start_idx = 0
    
    while start_idx + segment_length <= len(wav):
        segment = wav[start_idx:start_idx + segment_length]
        segments.append(segment)
        start_idx += hop_length
    
    logger.info(f"Split audio into {len(segments)} segments for analysis")
    
    # Process each segment
    predictions = []
    confidences = []
    segment_predictions = []
    
    for i, segment in enumerate(segments):
        try:
            # Check if segment is mostly silent
            segment_np = segment.numpy()
            segment_energy = np.mean(np.abs(segment_np))
            
            # Skip segments with very low energy (likely silent)
            if segment_energy < 0.001:
                logger.debug(f"Segment {i} has low energy ({segment_energy:.6f}), skipping")
                predictions.append(0.0)
                confidences.append(0.0)
                segment_predictions.append({
                    "segment": i + 1,
                    "class": "Not Snoring (Silent)",
                    "confidence": 0.0,
                    "probability": 0.0,
                    "energy": float(segment_energy)
                })
                continue
            
            # Ensure segment is exactly 16000 samples
            if len(segment) < segment_length:
                padding = segment_length - len(segment)
                segment = tf.pad(segment, [[0, padding]], mode='CONSTANT')
            
            # Create spectrogram
            spectrogram = create_spectrogram(segment)
            
            # Expand batch dimension
            spectrogram = tf.expand_dims(spectrogram, axis=0)
            
            # Predict
            prediction = model.predict(spectrogram, verbose=0)
            
            # Handle prediction output
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                prediction_value = float(prediction[0][0]) if len(prediction.shape) > 1 else float(prediction[0])
            else:
                prediction_value = float(prediction)
                
            predictions.append(prediction_value)
            confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value
            confidences.append(confidence)
            
            # Store segment prediction details
            segment_prediction = {
                "segment": i + 1,
                "class": "Snoring" if prediction_value > 0.5 else "Not Snoring",
                "confidence": float(confidence),
                "probability": float(prediction_value),
                "energy": float(segment_energy)
            }
            segment_predictions.append(segment_prediction)
            
        except Exception as e:
            logger.error(f"Error processing segment {i}: {e}")
            # Default to not snoring if prediction fails
            predictions.append(0.0)
            confidences.append(0.0)
            segment_predictions.append({
                "segment": i + 1,
                "class": "Not Snoring (Error)",
                "confidence": 0.0,
                "probability": 0.0
            })
    
    return wav, predictions, confidences, segment_predictions

def analyze_snoring_pattern(predictions, confidences, threshold=0.5):
    """Analyze the snoring pattern across segments"""
    snoring_segments = [1 if p > threshold else 0 for p in predictions]
    
    # Count snoring segments
    total_segments = len(snoring_segments)
    snoring_count = sum(snoring_segments)
    snoring_ratio = snoring_count / total_segments if total_segments > 0 else 0
    
    # Find snoring intervals (consecutive snoring segments)
    snoring_intervals = []
    current_interval = []
    
    for i, is_snoring in enumerate(snoring_segments):
        if is_snoring:
            current_interval.append(i)
        else:
            if len(current_interval) >= 2:  # At least 2 consecutive segments (1 second)
                start_time = current_interval[0] * 0.5  # Convert to seconds
                end_time = (current_interval[-1] + 1) * 0.5
                duration = end_time - start_time
                
                snoring_intervals.append({
                    "start_time": round(start_time, 2),
                    "end_time": round(end_time, 2),
                    "duration": round(duration, 2),
                    "segment_count": len(current_interval)
                })
            current_interval = []
    
    # Don't forget the last interval
    if len(current_interval) >= 2:
        start_time = current_interval[0] * 0.5
        end_time = (current_interval[-1] + 1) * 0.5
        duration = end_time - start_time
        
        snoring_intervals.append({
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(duration, 2),
            "segment_count": len(current_interval)
        })
    
    logger.info(f"Snoring Analysis: {total_segments} segments, {snoring_count} snoring ({snoring_ratio*100:.1f}%), {len(snoring_intervals)} intervals")
    
    return {
        "total_segments": total_segments,
        "snoring_segments": snoring_count,
        "snoring_ratio": float(snoring_ratio),
        "snoring_intervals": snoring_intervals,
        "interval_count": len(snoring_intervals)
    }, snoring_segments

def get_snoring_level_message(snoring_ratio: float) -> str:
    """Generate a user-friendly message based on snoring ratio"""
    if snoring_ratio > 0.3:
        return "🔊 HIGH snoring activity detected! Consider consulting a sleep specialist."
    elif snoring_ratio > 0.1:
        return "🔈 MODERATE snoring activity detected. Monitor your sleep patterns."
    else:
        return "🔇 LOW or no snoring detected. Your sleep appears to be quiet."

def generate_nudges(
    snoring_ratio: float,
    within_quiet_hours: bool
) -> Dict[str, Any]:
    """
    Generate non-intrusive night nudges based on snoring severity.
    """
    if snoring_ratio > 0.3 and within_quiet_hours:
        return {
            "nudges_this_hour": 1,
            "last_nudge_time": datetime.now().isoformat()
        }

    return {
        "nudges_this_hour": 0,
        "last_nudge_time": None
    }



@app.post("/analyze-snoring")
async def analyze_snoring(file: UploadFile = File(...)):
    """Main endpoint for snoring analysis"""
    try:
        # Validate file type - accept any audio file
        if not file.content_type or ("audio" not in file.content_type and 
                                     not any(file.filename.lower().endswith(ext) for ext in ['.mp3', '.m4a', '.wav', '.mp4', '.flac', '.aac', '.ogg', '.amr', '.webm'])):
            raise HTTPException(status_code=400, detail="Please upload an audio file (MP3, M4A, WAV, etc.)")
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        logger.info(f"Processing file: {file.filename}, Size: {len(file_content)} bytes")
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="AI model is not loaded. Please check server configuration.")
        
        # Convert to MP3 and remove silence
        mp3_content, original_duration = convert_to_mp3(file_content, file.filename)
        original_filename = file.filename
        
        # Check if audio is too short
        if len(mp3_content) < 1024:  # Less than 1KB
            raise HTTPException(status_code=400, detail="Audio file is too short or contains no sound after processing")
        
        # Process the audio
        wav, predictions, confidences, segment_predictions = preprocess_mp3_for_prediction(mp3_content, original_filename)
        
        # Analyze snoring pattern

        analysis_results, snoring_segments = analyze_snoring_pattern(predictions, confidences)




        # 🌙 Quiet hours evaluation (PATCH)
        quiet_start_str = NIGHT_PREFERENCES.get("quiet_start", "23:00")
        quiet_end_str = NIGHT_PREFERENCES.get("quiet_end", "07:00")
        quiet_start = datetime.strptime(quiet_start_str, "%H:%M").time()
        quiet_end = datetime.strptime(quiet_end_str, "%H:%M").time()
        now = datetime.now().time()
        within_quiet_hours = is_within_quiet_hours(now,quiet_start,quiet_end)


        # 🌙 Generate night nudges (PATCH)
        nudges = generate_nudges(analysis_results["snoring_ratio"],within_quiet_hours)


        
        # Calculate audio duration from tensor
        processed_duration = len(wav) / SAMPLE_RATE  # seconds


        # 💾 Save session to database
        db = SessionLocal()
        try:
            session = SnoringSession(
                audio_duration=processed_duration,
                snoring_ratio=analysis_results["snoring_ratio"],
                quiet_minutes=(1 - analysis_results["snoring_ratio"]) * processed_duration / 60,
                nudges_sent=nudges.get("nudges_this_hour", 0))
            db.add(session)
            db.commit()
            db.refresh(session)
            logger.info(f"📊 Session saved to DB with id={session.id}")
        finally:
            db.close()
        
        # Generate visualizations
        visualizations = create_visualizations(wav, predictions, snoring_segments, original_filename, processed_duration)
        
        # Generate conclusion message
        conclusion_message = get_snoring_level_message(analysis_results["snoring_ratio"])
        
        # Add audio quality information
        audio_info = {
            "original_duration": round(original_duration, 2),
            "processed_duration": round(processed_duration, 2),
            "silence_removed": original_duration - processed_duration > 1.0  # More than 1 second removed
        }
        
        # Prepare response
        response_data = {
            "status": "success",
            "filename": original_filename,
            "audio_duration": round(processed_duration, 2),
            "audio_info": audio_info,
            "analysis": analysis_results,
            "segment_predictions": segment_predictions,
            "message": conclusion_message,
            "summary": f"Snoring present in {analysis_results['snoring_ratio']*100:.1f}% of the audio",
            "nudges": nudges, 
            "visualizations": visualizations
        }
        
        logger.info(f"Analysis completed for {original_filename}: {analysis_results['snoring_ratio']*100:.1f}% snoring")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during snoring analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"model_loaded": False}
    
    try:
        return {
            "model_loaded": True,
            "model_layers": len(model.layers),
            "model_input_shape": str(model.input_shape),
            "model_output_shape": str(model.output_shape),
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {
            "model_loaded": True,
            "model_type": type(model).__name__,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)