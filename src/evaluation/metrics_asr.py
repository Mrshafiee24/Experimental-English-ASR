# Import libraries
import jiwer    # Calculate WER & CER

# Calculate CER(Character error rate)
def calculate_cer(reference: str | list[str], prediction: str | list[str]) -> float:
    return jiwer.cer(reference, prediction)

# Calculate WER(Word error rate)
def calculate_wer(reference: str | list[str], prediction: str | list[str]) -> float:
    return jiwer.wer(reference, prediction)