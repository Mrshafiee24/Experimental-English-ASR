# Import Librispeech
import csv
from pathlib import Path
import torch
import soundfile as sf
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Librispeech dataset
class LibriSpeechDataset(Dataset):
    def __init__(self, manifest_path):
        self.manifest_path= Path(manifest_path) 
        self.samples= []
        with open(self.manifest_path, "r", encoding= "utf-8") as file:
            reader= csv.DictReader(file)
            for row in reader:
                self.samples.append(row)
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample= self.samples[index]
        audio_path= sample["audio_path"]
        waveform, sample_rate= sf.read(audio_path, dtype= "float32")
        waveform= torch.tensor(waveform)
        
        return {
            "audio": waveform,
            "sample_rate": sample_rate,
            "transcript": sample["transcript"],
            "utterance_id": sample["utterance_id"],
            "speaker_id": sample["speaker_id"]
        }

# ASR Collator
class ASRCollator:
    def __init__(self, preprocessor, tokenizer):
        self.preprocessor= preprocessor
        self.tokenizer= tokenizer
    
    def __call__(self, batch):
        return collate_fn(batch, self.preprocessor, self.tokenizer)

# Collate function
def collate_fn(batch, preprocessor, tokenizer):
    features= []
    feature_lengths= []
    targets= []
    target_lengths= []
    transcripts= []
    utterance_ids= []
    speaker_ids= []

    for sample in batch:
        # Audio -> Log-mel
        feature = preprocessor(
            sample["audio"],
            original_sample_rate=sample["sample_rate"])
        features.append(feature)
        
        # Store actual feature length before padding
        feature_lengths.append(feature.shape[1])
        
        # Transcripts -> Token IDs
        target= torch.tensor(
            tokenizer.encode(sample["transcript"]),
            dtype= torch.long
        )
        targets.append(target)
        
        # Store actual target length
        target_lengths.append(target.shape[0])
        transcripts.append(sample["transcript"])
        utterance_ids.append(sample["utterance_id"])
        speaker_ids.append(sample["speaker_id"])
        
    # Pad features along the time dimension
    features= [
        feature.transpose(0, 1) # Swap dimensions from [80, T] to [T, 80]
        for feature in features
    ]
    
    padded_features= pad_sequence(
        features,
        batch_first= True
    )
    
    padded_features= padded_features.transpose(1, 2)
    
    # Pad targets
    padded_targets= pad_sequence(
        targets,
        batch_first= True,
        padding_value= 0
    )
        
    return {
        "features": padded_features,
        "feature_lengths": torch.tensor(feature_lengths, dtype= torch.long),
        "targets": padded_targets,
        "target_lengths": torch.tensor(target_lengths, dtype= torch.long),
        "transcripts": transcripts,
        "utterance_ids": utterance_ids,
        "speaker_ids": speaker_ids
    }

# Test
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    from src.preprocessing.audio_preprocessing import AudioPreprocessor
    from src.data.char_tokenizer import CharacterTokenizer

    # Configuration
    manifest_train_path= r"data\processed\manifests\train.csv"
    preprocessor= AudioPreprocessor()
    tokenizer= CharacterTokenizer()
    dataset= LibriSpeechDataset(manifest_train_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Test one sample
    sample= dataset[0]
    waveform= sample["audio"]
    transcript= sample["transcript"]
    
    features= preprocessor(waveform, original_sample_rate=sample["sample_rate"])
    targets= tokenizer.encode(transcript)
    
    print("\nSingle sample:")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Sample rate: {sample['sample_rate']}")
    print(f"Feature shape: {features.shape}")
    print(f"Target size: {len(targets)}")
    print(f"Transcript: {sample['transcript']}")
    print(f"Target: {targets}")
    print(f"Utterance id: {sample['utterance_id']}")
    
    # DataLoader
    loader= DataLoader(
        dataset,
        batch_size= 2,
        shuffle= False,
        collate_fn= lambda batch: collate_fn(
            batch,
            preprocessor,
            tokenizer
        )
    )
    
    # Get first batch
    batch= next(iter(loader))   # Get first batch
    
    # Report
    print("\n CTC Batch:")
    print(f"Features shape: {batch['features'].shape}")
    print(f"Feature lengths: {batch['feature_lengths']}")
    print(f"Targets shape: {batch['targets'].shape}")
    print(f"Target lengths: {batch['target_lengths']}")
    print(f"Transcripts: {batch['transcripts']}")
    print(f"Utterance ids: {batch['utterance_ids']}")
    
    