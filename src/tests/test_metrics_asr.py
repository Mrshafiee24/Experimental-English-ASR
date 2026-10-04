# Import libraries
from src.evaluation.metrics_asr import calculate_cer, calculate_wer
import unittest

# Test metrics
class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        reference= "THE CAT IS SITTING"
        prediction= "THE CAT IS SITTING"
        self.assertEqual(calculate_cer(reference, prediction), 0.0)
        self.assertEqual(calculate_wer(reference, prediction), 0.0)
    
    def test_word_substituation(self):
        reference= "THE CAT IS SITTING"
        prediction= "THE DOG IS SITTING"
        self.assertEqual(calculate_wer(reference, prediction), 0.25)
        
    def test_character_error(self):
        reference= "CAT"
        prediction= "BAT"
        self.assertEqual(calculate_cer(reference, prediction), 1/3)
        
# Test
if __name__== "__main__":
    unittest.main()
