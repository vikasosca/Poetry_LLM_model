from my_tokenizer import CharTokenizer, BPETokenizer
from model import EmbeddingLayer , SelfAttention , tinyLLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from datetime import datetime
import gc

# Load data & tokenizer
try:    
    with open('C:\\Users\\Admin\\Desktop\\AI\\python\\LLM\\CombinedInput.txt','r',encoding='utf-16') as f:
        text = f.read()
    #tokenizer = CharTokenizer(text)
        tokenizer = BPETokenizer()
        text = ''.join(char for char in text if ord(char) < 127 or char in '\n\t .,!?;:"')
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # Remove or replace invalid characters
    
except UnicodeDecodeError:
    raise ValueError("Could not decode file with any encoding")
  
# Load existing model or create new one
print(f'define checkpoints and start_step')
incremental_checkpoint_path = 'tinyllm_latest.pth' #for resuming training
checkpoint_path = 'tinyllm_checkpoint.pth'  # ← Use this for deployment
start_step = 0

# Train/Test split
n = int(0.85*len(data))
train_data = data[:n]
test_data = data[n:]

#Get Batch        
def get_batch(split, batch_size=64,block_size=32):
    data = train_data if split =='Train' else test_data
    ix = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x, y

# In case the model is being used for incremental training
if os.path.exists(incremental_checkpoint_path):
    print("Loading existing model for incremental training...")
    checkpoint = torch.load(incremental_checkpoint_path, map_location='cpu')
     
# Handle potential vocab size mismatch
    old_vocab_size = checkpoint['vocab_size']
    # Create tokenizer from current data
    new_vocab_size = tokenizer.vocab_size # len(tokenizer.chars)
    print(f"Old vocab: {old_vocab_size}, New vocab: {new_vocab_size}")       
    
    if new_vocab_size != old_vocab_size:
        print(f"Vocab size changed: {old_vocab_size} → {new_vocab_size}")
        model = tinyLLM(
            vocab_size=new_vocab_size,  # Use new vocab size if it changed
            embed_dim=checkpoint['embed_dim'],
            block_size=checkpoint['block_size'],
            num_blocks=checkpoint['num_blocks']
        )    
        model_dict = model.state_dict()
        pretrained_dict = {}
        
        pretrained_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict,strict=False)
        # DON'T load optimizer state - create fresh optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.1e-4)
        optimizer.zero_grad(set_to_none=True)
        print("Fresh optimizer created (vocab changed)")
    else:
        print('Vocab size Unchanged...')
        #tokenizer = CharTokenizer(text)  # Create tokenizer
        tokenizer = BPETokenizer()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        model = tinyLLM(
            vocab_size=new_vocab_size,
            embed_dim=checkpoint['embed_dim'],
            block_size=checkpoint['block_size'],
            num_blocks=checkpoint['num_blocks']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=2.1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded optimizer state (vocab unchanged)")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    start_step = checkpoint.get('step', 0)
    print(f'Resuming from step: {start_step}')

else:
# Initialize model
    print("Starting new model training...")
    print("No checkpoint found - starting from scratch")
    model = tinyLLM(vocab_size=tokenizer.vocab_size,  # ← BPETokenizer has 'vocab_size' property
                    embed_dim=768,                    # ← Increase for BPE
                    block_size=512,                   # ← Increase for longer context
                    num_blocks=12,                    # ← Increase for better learning 
                    dropout=0.1)
#Training setup
    optimizer = torch.optim.AdamW(model.parameters(),lr = 2.1e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    start_step = 0    
print(f"Training on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")   

def generate_greedy(model, tokenizer, start_text="", max_new_tokens=50):
    model.eval()
    idx = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            logits, _ = model(idx_cond)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            idx = torch.cat((idx, next_token), dim=1)
    return tokenizer.decode(idx[0].tolist())

# Training loop
def train_steps():
    model.train()
    
    xb,yb = get_batch('Train',batch_size=1,block_size=10)
    #print("Input :", repr(tokenizer.decode(xb[0].tolist())))
    #print("Target:", repr(tokenizer.decode(yb[0].tolist())))
    
    xb,yb = xb.to(device),yb.to(device)
    logits,loss = model(xb,targets=yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
# Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()   
    return loss.item()
# Validation loop
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    
    for split in ('Test','Train'):
        batch_losses = []
        for k in range(10):
            xb,yb = get_batch(split,batch_size=64,block_size=8)
            xb,yb = xb.to(device),yb.to(device)
            _, loss = model(xb, targets=yb)
            #logits, loss = model(xb, targets=yb)
            batch_losses.append(loss.item())
        out[split] = sum(batch_losses) / len(batch_losses)
        model.train()
    return out

# Train for a few epochs
print(f"Starting training with steps...{start_step}")
# Training loop
total_steps = start_step + 55000  # Train for 10k more steps on combined data
for step in range(total_steps):
    if step % 2000 == 0:
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        losses = estimate_loss()
        # Calculate Loss
        train_loss_val = losses['Train']
        test_loss_val = losses['Test']
        
        # Save checkpoint here with validation metrics
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': model.vocab_size,
            'embed_dim': model.embed_dim,
            'block_size': model.block_size,
            'num_blocks': model.num_blocks,
            'step': step,              # ← Current step
            'train_loss': train_loss_val,  # ← Current metrics
            'val_loss': test_loss_val,      # ← Current metrics
            'timestamp': datetime.now().isoformat()
        }
        
        # Save with metrics in filename
        torch.save(checkpoint, f'tinyllm_step_{step}_val_{losses["Test"]:.3f}.pth')
        print(f"💾 Checkpoint saved: tinyllm_step_{step}_Test_{losses['Test']:.3f}.pth")
        
        # Also save latest checkpoint
        torch.save(checkpoint, 'tinyllm_latest.pth')
        # Quick generation sample every 5k steps
        if step % 5000 == 0:
            sample = generate_greedy(model, tokenizer, "The ", max_new_tokens=30)
            print(f"Sample: {repr(sample)}")
    
    train_steps()
print("Training complete!")

# At the end of train.py (after training completes), we save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': model.vocab_size,
    'embed_dim': model.embed_dim,
    'block_size': model.block_size,
    'num_blocks': model.num_blocks,
}, checkpoint_path ) #'tinyllm_checkpoint.pth')

print("Model saved as: tinyllm_checkpoint.pth in python folder")

def generate_text(model,tokenizer,start_text="",max_new_tokens=100,temperature=0.5):
    #  Generate text using the trained model
    model.eval()
     # Encode starting text
    if start_text:
        idx = torch.tensor(tokenizer.encode(start_text),dtype=torch.long).unsqueeze(0)
    else:
        idx = torch.randint(0,tokenizer.vocab_size, (1,1))
    idx = idx.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
        # Get Logits
            logits, _ = model(idx_cond)
        # Focus on last token
            logits = logits[:,-1,:] / temperature # adjust for temperature
        # Top-k sampling to avoid repetition
            top_k = 50
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
         # Apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1)
        # Sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1)
        # Append sampled token
            idx = torch.cat( (idx,idx_next),dim=1)
        # Decode and return
        generated_text = tokenizer.decode(idx[0].tolist())
    return generated_text
    # Test generation after training
print("\nGenerating sample text...")
generated = generate_text(model,tokenizer,start_text="The",max_new_tokens=50)
print("Generated text:")
print(repr(generated))

# Greedy generation (no randomness)
print("\nGreedy generation:")
print(repr(generate_greedy(model, tokenizer, "To be", max_new_tokens=30)))
# Low-temp sampling
print("\nLow-temp sampling:")
print(repr(generate_text(model, tokenizer, "Once upon", max_new_tokens=30, temperature=0.7)))
print(repr(generate_text(model, tokenizer, "O Sita, my love, like a cloud wandering lonely", max_new_tokens=80)))