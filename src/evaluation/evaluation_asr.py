# Import libraries
import torch
import csv
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.Librispeech_dataset import LibriSpeechDataset, ASRCollator
from src.data.char_tokenizer import CharacterTokenizer
from src.preprocessing.audio_preprocessing import AudioPreprocessor
from src.models.asr_model import ASRModel
from src.decoding.greedy_decoder import greedy_ctc_decoder
from src.evaluation.metrics_asr import calculate_wer, calculate_cer

# Device
device= torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
    )
print(f"\nDevice: {device}")

# Path
manifest_test_path= Path("data/processed/manifests/test.csv")
checkpoint_path= Path("checkpoints/best_asr_model.pt")
result_path= Path("results/evaluation/test_predictions.csv")

# Component
test_dataset= LibriSpeechDataset(manifest_test_path)
preprocessor= AudioPreprocessor()
tokenizer= CharacterTokenizer()
collator= ASRCollator(preprocessor, tokenizer)

# Dataloader
test_loader= DataLoader(
    test_dataset,
    batch_size= 2,
    shuffle= False,
    num_workers= 0,
    collate_fn= collator
)

# Report
print("\n Evaluation dataset:")
print(f"Test: samples= {len(test_dataset)}, batches= {len(test_loader)}")

# Model
model= ASRModel(
    vocab_size= len(tokenizer.vocabulary)
).to(device)

# Load best checkpoint
checkpoint= torch.load(
    checkpoint_path,
    map_location= device,
    weights_only= True
)
model.load_state_dict(checkpoint)
print("\nBest model checkpoint loaded.")

# Evaluation
model.eval()
references= []
predictions= []
sample_results= []
with torch.no_grad():
    for batch_index, batch in enumerate(test_loader):
        # Move features to device
        features= batch["features"].to(device)
        feature_lengths= batch["feature_lengths"].to(device)
        
        # Forward pass
        logits= model(features)
        
        # Greedy CTC decoding
        decoded_texts= greedy_ctc_decoder(
            logits,
            tokenizer= tokenizer,
            blank_id= 0,
            input_length= feature_lengths
        )
        
        # Store references & predictions
        batch_references= batch["transcripts"]
        references.extend(batch["transcripts"]) # Extend== add at the last of list
        predictions.extend(decoded_texts)
        
        # Store per-sample results
        for index, (reference, prediction) in enumerate(zip(batch_references, decoded_texts)):
            sample_results.append({
                "utterance_id": batch["utterance_ids"][index],
                "speaker_id": batch["speaker_ids"][index],
                "reference": reference,
                "prediction": prediction,
                "cer": calculate_cer(reference, prediction),
                "wer": calculate_wer(reference, prediction),    
            })

        # Progress
        if (batch_index + 1) % 10 == 0:
            print(f"Evaluation batch: {batch_index + 1}/{len(test_loader)}")
            
# Calculate metrics
cer= calculate_cer(references, predictions)
wer= calculate_wer(references, predictions)

# Save per-sample results
result_path.parent.mkdir(parents= True, exist_ok= True)
with result_path.open(
    mode= "w",
    newline= "",
    encoding= "utf-8"
) as csv_file:
    fieldnames= [
        "utterance_id",
        "speaker_id",
        "reference",
        "prediction",
        "cer",
        "wer",
    ]
    writer= csv.DictWriter(
        csv_file,
        fieldnames= fieldnames
    )
    writer.writeheader()    # Write columns
    writer.writerows(sample_results)     # Write row's data

# Final report
print("\nASR Evaluation:")
print(f"Samples: {len(references)}")
print(f"CER: {cer:.4f}")
print(f"WER: {wer:.4f}")
print(f"Per-sample results saved to: {result_path}")