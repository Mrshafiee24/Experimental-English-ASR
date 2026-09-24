# Import libraries
import soundfile as sf
import csv
from pathlib import Path

# Configuration
dataset_dir= Path("data/raw/LibriSpeech")
output_dir= Path("data/processed/manifests")

splits= {
    "train": "train-clean-100",
    "dev": "dev-clean",
    "test": "test-clean"
}

# Build manifest
def build_manifest(split_name, dataset_name):
    # Build a csv manifest for one librispeech split
    # Define path
    split_dir= dataset_dir / dataset_name
    output_file= output_dir / f"{split_name}.csv"
    
    # Create output directory
    output_dir.mkdir(parents= True, exist_ok= True)
    
    # Store manifest row
    row= []
    
    # Find transcript file
    transcript_files= split_dir.rglob("*.trans.txt")    # find all transcript files
    for transcript_file in transcript_files:
        
        # Extract speaker and chapter id
        chapter_dir= transcript_file.parent
        speaker_id= chapter_dir.parent.name
        chapter_id= chapter_dir.name
        
        # Read transcripts
        with open(transcript_file, "r", encoding= "utf-8") as file:
            for line in file:
                line= line.strip()  # Remove unnecessry
                if not line:
                    continue
                
                # Split utterance id and transcript
                utterance_id, transcript= line.split(" ", maxsplit= 1)
                
                # Find audio file
                audio_file = chapter_dir / f"{utterance_id}.flac"
                if not audio_file.exists():
                    print(f"Warning: missing audio file: {audio_file}")
                    continue
                
                # Get audio duration
                info= sf.info(audio_file)
                duration= info.duration
                
                # Add manifest row
                row.append(
                    {
                        "audio_path": str(audio_file),
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "utterance_id": utterance_id,
                        "transcript": transcript,
                        "duration": duration,
                    }
                )        
        
    # Write CSV
    fieldnames = [
        "audio_path",
        "speaker_id",
        "chapter_id",
        "utterance_id",
        "transcript",
        "duration"
    ]
    with open(output_file, "w",newline= "", encoding= "utf-8") as file: # Open file
        writer= csv.DictWriter(file, fieldnames= fieldnames )   # Write dictionary to csv
        writer.writeheader()    # Write name of column
        writer.writerows(row)   # Write all data to rows
    
    # Report 
    print(f"{split_name}: {len(row)} samples")
    print(f"Saved: {output_file}")
    
# Main
if __name__== "__main__":
    for split_name, dataset_name in splits.items():
        build_manifest(split_name, dataset_name)