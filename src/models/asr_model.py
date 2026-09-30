# Import libraries
from torch.nn import (Module, Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU, LSTM, Linear)

# ASR model
class ASRModel(Module):
    def __init__(self, vocab_size= 29, hidden_size= 256):
        
        # Initialize the parent
        super().__init__()
        
        # CNN feature extrator
        self.cnn= Sequential(   # Run layers in order
            
            # Layer1
            Conv2d(in_channels= 1, out_channels= 32, kernel_size= 3, padding= 1), # Extract n feature maps
            BatchNorm2d(32), # Normalize the n feature maps
            ReLU(), # Add non-linearity
            MaxPool2d(kernel_size= (2, 1), stride= (2, 1)), # Reduce frequency dimension
            
            # Layer2
            Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3, padding= 1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size= (2, 1), stride= (2, 1))
            
        )
        
        # LSTM(Model temporal dependencies)
        self.lstm= LSTM(input_size= 64 * 20, hidden_size= hidden_size, num_layers= 2, batch_first= True, bidirectional= True, dropout= 0.1)
        
        # Bidirectional LSTM(Map BiLSTM features to vocabulary logits)
        self.classifier= Linear(hidden_size * 2, vocab_size)
        
    def forward(self, features):
        
        # Input
        x= features.unsqueeze(1)
        
        # Configuration
        x= self.cnn(x)
        x= x.permute(0, 3, 1, 2) # Relocation
        x= x.reshape(x.size(0), x.size(1), -1)
        x, _= self.lstm(x)
        logits= self.classifier(x)
        
        return logits
        
# Test
if __name__== "__main__":
    from torch.utils.data import DataLoader
    from src.data.Librispeech_dataset import LibriSpeechDataset, collate_fn
    from src.preprocessing.audio_preprocessing import AudioPreprocessor
    from src.data.char_tokenizer import CharacterTokenizer
    
    # Configuration
    manifest_train_path= r"data\processed\manifests\train.csv"
    preprocessor= AudioPreprocessor()
    tokenizer= CharacterTokenizer()
    dataset= LibriSpeechDataset(manifest_train_path)
    
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
    batch= next(iter(loader))
    features= batch["features"]
    model= ASRModel(vocab_size= len(tokenizer.vocabulary))
    logits= model(features)
    
    # Report
    print(f"\nModel test: \tInput= {features.shape}, \tOutput= {logits.shape}")