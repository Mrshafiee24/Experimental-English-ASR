# Import libraries
import csv
import jiwer
from pathlib import Path

# Path
result_path= Path("results/evaluation/test_predictions.csv")

# Load result
samples= []
with result_path.open(
    mode= "r",
    encoding= "utf-8"
) as csv_file:
    reader= csv.DictReader(csv_file)
    for row in reader:
        # Convert metric values from string to float
        row["cer"]= float(row["cer"])
        row["wer"]= float(row["wer"])
        samples.append(row)

# Basic statistic
total_samples= len(samples)
exact_matches= sum(
    sample["reference"]== sample["prediction"]
    for sample in samples
)

average_cer= sum(
    float(sample["cer"])
    for sample in samples
)/ total_samples

average_wer= sum(
    float(sample["wer"])
    for sample in samples
)/ total_samples

# Error type analysis
total_substitutions=0
total_delections= 0
total_insertions= 0
for sample in samples:
    reference= sample["reference"]
    prediction= sample["prediction"]
    output= jiwer.process_words(reference, prediction)
    total_substitutions+= output.substitutions
    total_delections+= output.deletions 
    total_insertions+= output.insertions 

total_errors= (
    total_substitutions
    + total_delections
    + total_insertions
)
total_reference_words= sum(
    len(sample["reference"].split())
    for sample in samples
)

# Error rate
substitution_rate= (total_substitutions / total_reference_words)
delection_rate= (total_delections / total_reference_words)
insertion_rate= (total_insertions / total_reference_words)

# Sort samples by error
worst_cer_samples= sorted(
    samples,
    key= lambda sample: sample["cer"],
    reverse= True
)
worst_wer_samples= sorted(
    samples,
    key= lambda sample: sample["wer"],
    reverse= True
)

# Report
print("\nASR error analysis:\n")
print(f"Total samples: {total_samples}")
print(f"Exact matches: {exact_matches}")
print(f"Exact matches rate: {exact_matches / total_samples:.4f}")
print(f"Average CER: {average_cer}")
print(f"Average WER: {average_wer}")

print("Error type analysis:")
print(f"Substitutions: {total_substitutions}")
print(f"Delections: {total_delections}")
print(f"Insertions: {total_insertions}")
print(f"Substitution rate: {substitution_rate:.4f}")
print(f"Delection rate: {delection_rate:.4f}")
print(f"Insertion rate: {insertion_rate:.4f}")

# Top CER error
print("\nTop 10 highes CER:")
for rank, sample in enumerate(worst_cer_samples[: 10], start= 1):
    print(f"\n[{rank}]")
    print(f"Utterance id: {sample['utterance_id']}")
    print(f"Speaker id: {sample['speaker_id']}")
    print(f"CER: {sample['cer']:.4f}")
    print(f"WER: {sample['wer']:.4f}")
    print(f"Reference: {sample['reference']}")
    print(f"prediction: {sample['prediction']}")

# Top WER error
print("\nTop 10 highes WER:")
for rank, sample in enumerate(worst_wer_samples[: 10], start= 1):
    print(f"\n[{rank}]")
    print(f"Utterance id: {sample['utterance_id']}")
    print(f"Speaker id: {sample['speaker_id']}")
    print(f"WER: {sample['wer']:.4f}")
    print(f"CER: {sample['cer']:.4f}")
    print(f"Reference: {sample['reference']}")
    print(f"prediction: {sample['prediction']}")    