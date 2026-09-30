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
        
    def __call__(self, waveform):
        if waveform.ndim== 2:
            waveform= waveform.squeeze(0)
        
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