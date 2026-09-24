# Import libraries
import soundfile as sf
import csv
from pathlib import Path

# Configuration
manifast_dir= Path("data/processed/manifests")
splits= ["train", "dev", "test"]
expected_sr= 16000
expected_channel= 1

# Validation
def validation_manifest(split_name):
    # validate on LibriSpeech manifest
    
    manifest_file= manifast_dir / f"{split_name}.csv"
    print(f"\n{split_name} validation...")
    
    # Counter
    total_sample= 0
    missing_audio= 0
    empty_transcripts= 0
    invalid_duration= 0
    invalid_sample_rate= 0
    invalid_channel= 0
    
    # Store IDs for duplicate detection
    utterance_ids= set()
    duplicate_ids= 0
    
    # Read manifest
    with open(manifest_file, "r", encoding= "utf-8") as file:
        reader= csv.DictReader(file)
        for row in reader:
            total_sample+= 1
            
            # Get information
            audio_path= Path(row["audio_path"])
            transcript= row["transcript"].strip()
            utterance_id= row["utterance_id"]
            duration= float(row["duration"])
            
            # Check audio file
            if not audio_path.exists():
                missing_audio+= 1
                continue
            
            # Check transcript
            if not transcript:
                empty_transcripts+= 1
            
            # Check duration
            if duration <= 0:
                invalid_duration+= 1
            
            # Check duplicate utterance id
            if utterance_id in utterance_ids:
                duplicate_ids+= 1
            else:
                utterance_ids.add(utterance_id)
            
            # Read audio metadata
            info= sf.info(audio_path)
            
            # Check sample rate
            if info.samplerate != expected_sr:
                invalid_sample_rate+= 1
                
            # Check channel
            if info.channels != expected_channel:
                invalid_channel+= 1
            
    # Report
    print(f"\nTotal sample: {total_sample}")
    print(f"\nMissing audio: {missing_audio}")
    print(f"\nEmpty transcripts: {empty_transcripts}")          
    print(f"\nDuplicated ids: {duplicate_ids}")  
    print(f"\nInvalid duration: {invalid_duration}")
    print(f"\nInvalid sample_rate: {invalid_sample_rate}")
    print(f"\nInvalid invalid_channel: {invalid_channel}")
    
    # Final status
    total_errors= (
        missing_audio
        + empty_transcripts
        + duplicate_ids
        + invalid_duration
        + invalid_sample_rate
        + invalid_channel
    )
    if total_errors == 0:
        print("\nStatus: Pass")
    else:
        print("\nStatus: Fail")
    
# Main
if __name__== "__main__":
    for split in splits:
        validation_manifest(split)
                

