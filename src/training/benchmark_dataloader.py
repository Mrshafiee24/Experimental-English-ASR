# Import libraries
import time
import torch
from torch.utils.data import DataLoader
from src.data.Librispeech_dataset import LibriSpeechDataset, ASRCollator
from src.data.char_tokenizer import CharacterTokenizer
from src.preprocessing.audio_preprocessing import AudioPreprocessor

def main():
    # Path
    manifest_train_path= r"data\processed\manifests\train.csv"

    # Component
    train_dataset= LibriSpeechDataset(manifest_train_path)
    preprocessor= AudioPreprocessor()
    tokenizer= CharacterTokenizer()

    # Dataloader
    collator= ASRCollator(preprocessor, tokenizer)
    train_loader= DataLoader(
        train_dataset,
        batch_size= 4,
        shuffle= True,
        num_workers= 0, # 2 -> 0, Faster
        collate_fn= collator
    )

    # Benchmark settings
    num_batches= 50
    previous_time= 0
    batch_times= []

    # Benchmark
    print("\nStart dataloader benchmark...")
    start_total= time.perf_counter()
    for batch_index, batch in enumerate(train_loader):
        if batch_index >= num_batches:
            break
        
        current_time= time.perf_counter()

        if batch_index > 0:
            batch_time= current_time - previous_time
            batch_times.append(batch_time)
        
            print(
                f"Batch {batch_index:02d}/{num_batches} | "
                f"Load time: {batch_time:4f} sec"
            )
        previous_time= current_time

    end_total= time.perf_counter()
    total_time= end_total - start_total
    average_time= sum(batch_times) / len(batch_times)

    # Report
    print("\nDataloader benchmark:")
    print(f"Total batches: {len(batch_times)}")
    print(f"Total time: {total_time:4f} sec")
    print(f"Average batch load time: {average_time:4f} sec")    
    print(f"Batches per second: {1 / average_time:2f}")
    
# Test
if __name__== "__main__":
    main()