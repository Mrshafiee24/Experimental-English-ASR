# Import libraries
import torch
import torchaudio

# Configuration
sample_rate= 16000
n_fft= 400
hop_length= 160
n_mels= 80

# Audio preprocessing
class AudioPreprocessor:
    def __init__(self, sample_rate= 16000, n_fft= 400, hop_length= 160, n_mels= 80):
        self.sample_rate= sample_rate
        self.mel_transform= torchaudio.transforms.MelSpectrogram(
            sample_rate= sample_rate,
            n_fft= n_fft,
            hop_length= hop_length,
            n_mels= n_mels,
            power= 2.0
        )
        self.amplitude_to_db= torchaudio.transforms.AmplitudeToDB()
        
    def __call__(self, waveform, original_sample_rate=16000):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
        
        # Convert stereo audio to mono
        if waveform.ndim == 2:
            if waveform.shape[0] <= 2:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.mean(dim=1)
        
        # Resample when the original rate differs
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=self.sample_rate
            )
            
        mel= self.mel_transform(waveform)
        log_mel= self.amplitude_to_db(mel)
        
        return log_mel
    
# Test
if __name__ == "__main__":
    from src.data.Librispeech_dataset import LibriSpeechDataset

    # Configuration
    manifest_train_path= r"data\processed\manifests\train.csv"
    
    dataset= LibriSpeechDataset(manifest_train_path)
    sample= dataset[0]
    waveform= sample["audio"]
    
    print(f"\nWaveform: \nshape: {waveform.shape},\n sample rate: {sample['sample_rate']}")
    
    # Preprocessor
    preprocessor= AudioPreprocessor()
    log_mel= preprocessor(waveform)
    
    print(f"\nLog-mel: \nshape: {log_mel.shape},\n min: {log_mel.min().item()},\n max: {log_mel.max().item()} ")