# Import libraries
import torch
import torch.nn as nn   # Pytorch neural network modules

# Configuration
blank_id= 0
vocab_size= 4

# Target: A B C
targets= torch.tensor([[1, 2, 3]], dtype= torch.long)

# Model outputs: (raw output)
# Time steps= 6, Batch size= 1, Vocabulary size= 4
# Expected best path= A A blank B B C

logits= torch.tensor(
    [ 
       [
        [5.0, 0.0, 0.0, 0.0], # A
        ],
       [
        [5.0, 0.0, 0.0, 0.0], # A
       ],
       [
        [5.0, 4.0, 0.0, 0.0], # blank
       ],
       [
        [5.0, 0.0, 4.0, 0.0], # B
       ],
       [
        [5.0, 0.0, 4.0, 0.0], # B
       ],
       [
        [5.0, 0.0, 0.0, 4.0]  # C
       ]
    ],
    dtype= torch.float32
)

# Convert logits to log probablities(Now have explanation of probability)
log_probs= torch.log_softmax(
    logits,
    dim= -1
)

# Input sequence length
input_lengths= torch.tensor(
    [6],
    dtype= torch.long
)

# Target sequence length
target_lengths= torch.tensor(
    [3],
    dtype= torch.long
)

# CTC loss(How well dose the model's prediction match the actual text)
ctc_loss= nn.CTCLoss(
    blank= blank_id,
    zero_infinity= True
)
# Compute ctc loss
loss= ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths
)

# Report
print(f"\nLogits shape: {logits.shape}")
print(f"Log probablities shape: {log_probs.shape}")
print(f"Targets: {targets}")
print(f"Input lengths: {input_lengths}")
print(f"Target lengths: {target_lengths}")
print(f"CTC loss: {loss.item()}")
