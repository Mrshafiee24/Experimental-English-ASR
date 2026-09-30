# Character tokenizer
class CharacterTokenizer:
    def __init__(self):
        self.vocabulary= [
            "<blank>",
            " ",
            "'",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z"
        ]
        
        self.char_to_id= {
            char: index
            for index, char in enumerate(self.vocabulary)   # Give each character one number and convert to dictionary
        }
        self.id_to_char= {
            index: char
            for index, char in enumerate(self.vocabulary)
        }
        
    def encode(self, text):
        text= text.upper()
        token_ids= [
            self.char_to_id[char]
            for char in text
            ]
        return token_ids
        
    def decode(self, token_ids):
        chars= [
            self.id_to_char[token_id]
            for token_id in token_ids
        ]
        return "".join(chars)
    
# Test
if __name__== "__main__":
    # Configuration
    tokenizer= CharacterTokenizer()
    text= "Hello"
    encoded= tokenizer.encode(text)
    decoded= tokenizer.decode(encoded)
    
    # Report
    print("\nVocabulary:")
    print(f"Orginal: {text}")
    print(f"Size: {len(tokenizer.vocabulary)}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")