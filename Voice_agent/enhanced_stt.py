"""
Enhanced Speech-to-Text Module
Combines all free optimization strategies for better Arabic transcription
"""

import whisper
import numpy as np
import soundfile as sf
import tempfile
import os
import re
try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False
    print("⚠️  noisereduce not installed. Install with: pip install noisereduce")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EnhancedSTT:
    """
    Enhanced Speech-to-Text with multiple free optimizations:
    - Larger model support (medium/large)
    - Audio preprocessing (noise reduction, normalization)
    - Optimized Whisper parameters
    - Text post-processing
    """
    
    def __init__(self, model_size="medium", enable_noise_reduction=True):
        """
        Initialize Enhanced STT
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       Recommendation: medium for best accuracy/speed balance
            enable_noise_reduction: Enable audio preprocessing (requires noisereduce)
        """
        print(f"🔧 Loading Enhanced STT (Whisper {model_size})...")
        
        self.model_size = model_size
        self.model = whisper.load_model(model_size)
        self.sample_rate = 16000
        self.enable_noise_reduction = enable_noise_reduction and NOISE_REDUCTION_AVAILABLE
        
        if enable_noise_reduction and not NOISE_REDUCTION_AVAILABLE:
            print("⚠️  Noise reduction disabled (noisereduce not installed)")
        
        print(f"✓ Enhanced STT ready (model: {model_size}, "
              f"noise reduction: {self.enable_noise_reduction})")
    
    def preprocess_audio(self, audio):
        """
        Clean and enhance audio for better transcription
        
        Args:
            audio: numpy array of audio samples
        
        Returns:
            Preprocessed audio
        """
        # 1. Trim silence from beginning and end
        audio = self._trim_silence(audio)
        
        # 2. Normalize volume
        audio = self._normalize_volume(audio)
        
        # 3. Reduce noise (if enabled)
        if self.enable_noise_reduction:
            audio = self._reduce_noise(audio)
        
        return audio
    
    def _trim_silence(self, audio, threshold=0.01):
        """Remove silence from start and end"""
        non_silent = np.abs(audio) > threshold
        
        if non_silent.any():
            start_idx = np.argmax(non_silent)
            end_idx = len(audio) - np.argmax(non_silent[::-1])
            audio = audio[start_idx:end_idx]
        
        return audio
    
    def _normalize_volume(self, audio, target_rms=0.1):
        """Normalize audio to consistent volume"""
        current_rms = np.sqrt(np.mean(audio ** 2))
        
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        
        # Clip to prevent distortion
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _reduce_noise(self, audio):
        """Reduce background noise using spectral gating"""
        try:
            # Use first 0.5 seconds as noise profile
            noise_sample_duration = min(0.5, len(audio) / self.sample_rate / 2)
            noise_len = int(noise_sample_duration * self.sample_rate)
            
            cleaned_audio = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=True,
                prop_decrease=0.8  # Less aggressive = more natural
            )
            return cleaned_audio
        except Exception as e:
            print(f"⚠️  Noise reduction failed: {e}")
            return audio
    
    def transcribe(self, audio_data):
        """
        Transcribe audio with all enhancements
        
        Args:
            audio_data: numpy array of audio samples (float32, mono, 16kHz)
        
        Returns:
            str: Transcribed text in Arabic
        """
        # Preprocess audio
        audio_data = self.preprocess_audio(audio_data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio_data, self.sample_rate)
            temp_path = f.name
        
        try:
            # Transcribe with optimized parameters
            result = self.model.transcribe(
                temp_path,
                language='ar',  # Force Arabic
                task='transcribe',  # Not translate
                
                # Accuracy improvements
                beam_size=5,  # Wider beam search
                best_of=5,  # Consider more candidates
                temperature=0.0,  # Deterministic (no randomness)
                
                # Context and quality control
                condition_on_previous_text=True,  # Use context
                compression_ratio_threshold=2.4,  # Detect hallucinations
                logprob_threshold=-1.0,  # Confidence filtering
                no_speech_threshold=0.6,  # Silence detection
                
                # Performance
                fp16=TORCH_AVAILABLE and torch.cuda.is_available(),
            )
            
            text = result['text'].strip()
            
            # Post-process text
            text = self._clean_text(text)
            
            return text
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _clean_text(self, text):
        """
        Post-process transcribed text
        - Remove diacritics
        - Normalize characters
        - Fix spacing
        """
        # Remove Arabic diacritics (tashkeel)
        text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
        
        # Normalize common character variations
        text = re.sub('[إأآا]', 'ا', text)  # Alef variations
        text = re.sub('ى', 'ي', text)  # Alef maksura to ya
        text = re.sub('ؤ', 'و', text)  # Waw with hamza
        text = re.sub('ئ', 'ي', text)  # Ya with hamza
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\s+([،؛؟!.])', r'\1', text)  # Remove space before punctuation
        
        return text.strip()
    
    def get_model_info(self):
        """Get information about loaded model"""
        return {
            'model_size': self.model_size,
            'noise_reduction': self.enable_noise_reduction,
            'sample_rate': self.sample_rate
        }


# Convenience function for quick usage
def create_enhanced_stt(model_size="medium"):
    """
    Create enhanced STT instance
    
    Args:
        model_size: tiny, base, small, medium, large
                   Recommended: medium (best balance)
    
    Returns:
        EnhancedSTT instance
    """
    return EnhancedSTT(model_size=model_size)


# Example usage
if __name__ == "__main__":
    # Test the enhanced STT
    stt = create_enhanced_stt("base")
    
    print("\nModel info:")
    print(stt.get_model_info())
    
    print("\n✓ Enhanced STT is ready to use!")
    print("\nUsage:")
    print("  stt = EnhancedSTT('medium')")
    print("  text = stt.transcribe(audio_data)")
