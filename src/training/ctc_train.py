# Import libraries
import torch
import torch.nn as nn 
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.Librispeech_dataset import LibriSpeechDataset, ASRCollator
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
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_path= checkpoint_dir / "best_asr_model.pt"
resume_checkpoint_path= checkpoint_dir / "last_asr_checkpoint.pt"

# Component
train_dataset= LibriSpeechDataset(manifest_train_path)
dev_dataset= LibriSpeechDataset(manifest_dev_path)
preprocessor= AudioPreprocessor()
tokenizer= CharacterTokenizer()

# Dataloader(Get dataset's samples and give in the form of batch to model)
collator= ASRCollator(preprocessor, tokenizer)
train_loader= DataLoader(
    train_dataset,
    batch_size= 4,
    shuffle= True,
    num_workers= 0,
    collate_fn= collator
)
dev_loader= DataLoader(
    dev_dataset,
    batch_size= 2,
    shuffle= False,
    collate_fn= collator
)
print(f"\nDataset:")
print(f"Train samples: {len(train_dataset)}")
print(f"Train batches: {len(train_loader)}")
print(f"Validation samples: {len(dev_dataset)}")
print(f"Validation batches: {len(dev_loader)}")

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
start_epoch= 0
best_val_loss= float("inf") # First value placing true
# # Sanity check limits
# max_train_batches= 100
# max_val_batches= 20

# Resume model weights if checkpoint exists
checkpoint_path= "checkpoints/best_asr_model.pt"
if resume_checkpoint_path.exists():
    checkpoint= torch.load(
        resume_checkpoint_path,
        map_location= device,
        weights_only= True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch= checkpoint["epoch"]
    best_val_loss= checkpoint["best_val_loss"]
    
    print(f"Resuming from epoch {start_epoch}")
    
# Otherwise, load legacy model weights if available
elif best_checkpoint_path.exists():
    checkpoint= torch.load(
        best_checkpoint_path,
        map_location= device,
        weights_only= True
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    print("Loding existing model weights.")
    print("Optimizer and epoch state will start fresh.")
    
else:
    print("No checkpoint found, Training from scratch.")

# Training loop
for epoch in range(start_epoch, num_epochs):
    
    # Training
    model.train()
    train_loss= 0.0
    for batch_index, batch in enumerate(train_loader):
        # # Stop after limited number of batches
        # if batch_index >= max_train_batches:
        #     break
        
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
        
        # Progress
        if (batch_index + 1) % 10 == 0:
            # print(f"Train batch: {batch_index + 1}/{max_train_batches} | Loss: {loss.item():.4f}")
            print(f"Train batch: {batch_index + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
    # train_loss/= min(max_train_batches, len(train_loader))
    train_loss/= len(train_loader)

    # Validation
    model.eval()
    val_loss= 0.0
    with torch.no_grad(): # Turn off gradient on this part
        for batch_index, batch in enumerate(dev_loader):
            # # Stop after limited number of batches
            # if batch_index >= max_val_batches:
            #     break
        
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
            
            # Progress
            if (batch_index + 1) % 10 == 0:
                # print(f"Validation batch: {batch_index + 1}/{max_val_batches} | Loss: {loss.item():.4f}")
                print(f"Validation batch: {batch_index + 1}/{len(dev_loader)} | Loss: {loss.item():.4f}")
                
        # val_loss/= min(max_val_batches, len(dev_loader))
        val_loss/= len(dev_loader)

        # Report
        print(f"\nEpoch {epoch + 1} / {num_epochs}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss: {val_loss:.4f}")

        # Save the latest training state
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": min(best_val_loss, val_loss),
            },
            resume_checkpoint_path
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss= val_loss
            torch.save(
                model.state_dict(),
                best_checkpoint_path
            )
            print("\nBest checkpoint saved.")