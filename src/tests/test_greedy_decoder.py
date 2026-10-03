# Import libraries
import unittest     # Test part of code
import torch
from src.data.char_tokenizer import CharacterTokenizer
from src.decoding.greedy_decoder import greedy_ctc_decoder

class TestGreedyCTCDecoder(unittest.TestCase):
    def setUp(self):
        self.tokenizer= CharacterTokenizer()
        self.blank_id= self.tokenizer.char_to_id["<blank>"]
        self.a_id= self.tokenizer.char_to_id["A"]
        self.b_id= self.tokenizer.char_to_id["B"]
    
    def test_basic_ctc_decoding(self):
        # CTC best path: A A blank A blank B B
        predicted_ids= [
            self.a_id,
            self.a_id,
            self.blank_id,
            self.a_id,
            self.blank_id,
            self.b_id,
            self.b_id,
        ]
        
        # Create artificial logits: shape [B, T, V]
        logits= torch.full(
            (1, len(predicted_ids), len(self.tokenizer.vocabulary)),
            -10.0
            )
        
        # Make the expected token the highest - probability token
        for time_index, token_id in enumerate(predicted_ids):
            logits[0, time_index, token_id]= 10.0
            
        decoded= greedy_ctc_decoder(
            logits= logits,
            tokenizer= self.tokenizer,
            blank_id= self.blank_id
        )
        self.assertEqual(decoded, ["AAB"])
        
    def test_input_length(self):
        # Valid sequence: (A A blank B B blank blank)
        predicted_ids= [
            self.a_id,
            self.a_id,
            self.blank_id,
            self.b_id,
            self.b_id,            
            self.blank_id,
            self.blank_id,
        ]
        
        # Create artificial logits: shape [B, T, V]
        logits= torch.full(
            (1, len(predicted_ids), len(self.tokenizer.vocabulary)),
            -10.0
            )
        
        # Make the expected token the highest - probability token
        for time_index, token_id in enumerate(predicted_ids):
            logits[0, time_index, token_id]= 10.0
        
        input_lengths= torch.tensor([5], dtype= torch.long)
        decoded= greedy_ctc_decoder(
            logits= logits,
            tokenizer= self.tokenizer,
            blank_id= self.blank_id,
            input_length= input_lengths
        )
        
        self.assertEqual(decoded, ["AB"])   # Check actual value equal with expected value

if __name__== "__main__":
    unittest.main()