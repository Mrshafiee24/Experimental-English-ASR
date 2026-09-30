# Import libraries
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from src.data.Librispeech_dataset import LibriSpeechDataset, collate_fn
from src.data.char_tokenizer import CharacterTokenizer
from src.preprocessing.audio_preprocessing import AudioPreprocessor
from src.models.asr_model import ASRModel

# Device
device= torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"\nDevice: {device}")

# Path
manifest_train_path= r"data\processed\manifests\train.csv"
manifest_dev_path= r"data\processed\manifests\dev.csv"

# Component
train_dataset= LibriSpeechDataset(manifest_train_path)
dev_dataset= LibriSpeechDataset(manifest_dev_path)
preprocessor= AudioPreprocessor()
tokenizer= CharacterTokenizer()

# Dataloader(Get dataset's samples and give in the form of batch to model)
train_loader= DataLoader(
    train_dataset,
    batch_size= 2,
    shuffle= True,
    collate_fn= lambda batch: collate_fn(
        batch,
        preprocessor,
        tokenizer
    )
)
dev_loader= DataLoader(
    dev_dataset,
    batch_size= 2,
    shuffle= False,
    collate_fn= lambda batch: collate_fn(
        batch,
        preprocessor,
        tokenizer
    )
)
print(f"\nDataset: Train sample= {len(train_loader)}, Validation sample= {len(dev_loader)}")

# Model
model= ASRModel(vocab_size= len(tokenizer.vocabulary)).to(device)

# CTC Loss
ctc_loss= nn.CTCLoss(
    blank= 0,
    zero_infinity= True
)

# Optimizer
optimizer= torch.optim.Adam(
    model.parameters(),
    lr= 1e-3
)

# Training setting
num_epochs= 5
best_val_loss= float("inf") # First value placing true

# Training loop
for epoch in range(num_epochs):
    
    # Training
    model.train()
    train_loss= 0.0
    for batch in train_loader:
        features= batch["features"].to(device)
        targets= batch["targets"].to(device)
        feature_lengths= batch["feature_lengths"].to(device)
        target_lengths= batch["target_lengths"].to(device)
        
        # Clear previous gradients
        optimizer.zero_grad() # Clear
        
        # Forward pass
        logits= model(features)
        
        # Convert logits to probabilites
        log_probs= torch.log_softmax(logits, dim= -1)   # Dim== -1: latest dimention
        
        # CTC Expects [T, B, C]
        log_probs= log_probs.permute(1, 0, 2)
        
        # Calculate ctc loss
        loss= ctc_loss(
            log_probs,
            targets,
            feature_lengths,
            target_lengths
        )
        
        # Backward pass
        loss.backward()
        
        # Update model parameters(model weigth)
        optimizer.step()
        
        train_loss+= loss.item()
    train_loss/= len(train_loader)

    # Validation
    model.eval()
    val_loss= 0.0
    with torch.no_grad(): # Turn off gradient on this part
        for batch in dev_loader:
            features= batch["features"].to(device)
            targets= batch["targets"].to(device)
            feature_lengths= batch["feature_lengths"].to(device)
            target_lengths= batch["target_lengths"].to(device)       
            
            # Forward pass
            logits= model(features)
            
            # Convert logits to probabilites
            log_probs= torch.log_softmax(logits, dim= -1)   # Dim== -1: latest dimention
            
            # CTC Expects [T, B, C]
            log_probs= log_probs.permute(1, 0, 2)
            
            # Calculate ctc loss
            loss= ctc_loss(
                log_probs,
                targets,
                feature_lengths,
                target_lengths
            )
            
            val_loss+= loss.item()
        val_loss/= len(dev_loader)

        # Report
        print(f"\nEpoch {epoch + 1} / {num_epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss= val_loss
            torch.save(
                model.state_dict(),
                "checkpoints/best_asr_model.pt"
            )
            print("\nBest model saved.")