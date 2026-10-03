# Import libraries
import torch

# CTC Decorder
def greedy_ctc_decoder(
    logits: torch.Tensor,
    tokenizer,
    blank_id: int= 0,
    input_length= None,
) -> list[str]:
    
    # Validate input shape
    if logits.ndim != 3:
        raise ValueError("Logits must have shape [B, T, C].")
    
    # Select the most probable token at each time step
    predicted_ids= logits.argmax(dim= -1)   # dim= -1 == do oparation on last dimention
    
    # Use the full sequence length when input length is not provided
    if input_length is None:
        lengths= [predicted_ids.size(1)]* predicted_ids.size(0)
        
    # Convert Tensor length to Python list
    elif isinstance(input_length, torch.Tensor):    # isinstance == Check condition
        lengths= input_length.detach().cpu().tolist()
    
    else:
        lengths= list(input_length)
        
    # Validate number of length
    if len(lengths) != predicted_ids.size(0):
        raise ValueError("Input lengths must contain one length per sample")
    
    # Decode each sample independently
    decoded_texts= []
    for sequence, length in zip(predicted_ids, lengths):    # pairs together
        # Remove padded time steps
        token_ids= (sequence[: int(length)].detach().cpu().tolist())
        
        collapsed_ids= []
        previous_ids= None
        
        for token_id in token_ids:
            # Remmove consecutive duplicate and CTC blank tokens
            if (token_id != previous_ids and token_id != blank_id):
                collapsed_ids.append(token_id)
            
            previous_ids= token_id
        
        # Convert token IDs back to text
        text= tokenizer.decode(collapsed_ids)
        decoded_texts.append(text)
        
    return decoded_texts
