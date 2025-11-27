import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size,embed_dim)
        self.position_embedding = nn.Embedding(block_size,embed_dim)
        
    def forward(self, idx):
        """
        idx: (batch_size, block_size) tensor of token IDs
        Returns: (batch_size, block_size, embed_dim) embeddings
        """
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(idx.size(1), device=idx.device))  # (T, C)
        return tok_emb + pos_emb
    
class SelfAttention(nn.Module):
        def __init__(self, embed_dim,block_size,dropout=0.1):
            super().__init__()
            assert embed_dim%1 == 0
            self.embed_dim = embed_dim
            self.block_size = block_size
    # Projections for Q, K, V (single head)
            self.q_proj = nn.Linear(embed_dim,embed_dim)
            self.k_proj = nn.Linear(embed_dim,embed_dim)
            self.v_proj = nn.Linear(embed_dim,embed_dim)
            self.dropout = nn.Dropout(dropout)
    # Register causal mask (upper triangular = -inf)
            mask = torch.tril(torch.ones(block_size,block_size))
            mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('causal_mask',mask)
            print('Mask: ',mask)
        
        def forward(self,x):
        
            '''
            x: (batch_size, block_size, embed_dim)
            Returns: (batch_size, block_size, embed_dim)
            '''    
            B,T,C = x.shape
        # Project to Q, K, V
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        # Compute attention scores: (B, T, T)
            att =  (q @ k.transpose(-2, -1)) * (1.0 / (C ** 0.5))  # scaled dot-product
        # Apply causal mask (only attend to past & current tokens)
            att = att + self.causal_mask[:T, :T]  # crop mask to current sequence length
        # Softmax to get weights
            att = torch.softmax(att, dim =-1)
            att = self.dropout(att)
        # Weighted sum of values
            out = att @ v  # (B, T, C)
            return out

class MLP(nn.Module):
    def __init__(self, embed_dim,dropout=0.1):
         super().__init__()
         self.net = nn.Sequential(
             nn.Linear(embed_dim,4*embed_dim),nn.GELU() ,nn.Linear(4*embed_dim,embed_dim),
             nn.Dropout(dropout)
            )
    def forward(self,x):
        return self.net(x)   

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim,block_size, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim,block_size,dropout)
        self.mlp = MLP(embed_dim,dropout)
    #Layer Normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self,x):
    # Attention block: residual connection + layer norm
        x = x + self.attention(self.ln1(x))  # post-norm (applied before attention)
    # MLP block: residual connection + layer norm
        x = x + self.mlp(self.ln2(x)) # post-norm (applied before MLP)
        return x
    
    '''
    Build the Full TinyLLM Model
    '''
    
class tinyLLM(nn.Module):
        def __init__(self, vocab_size, embed_dim, block_size, num_blocks, dropout=0.1):
            super().__init__()
            self.block_size = block_size
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.block_size = block_size
            self.num_blocks = num_blocks
        # Token + Position embeddings
            self.embedding = EmbeddingLayer(vocab_size, embed_dim, block_size)
        # Stack of transformer blocks
            self.blocks = nn.Sequential(*[
                         TransformerBlock(embed_dim, block_size, dropout)
                         for _ in range(num_blocks)
                        ])
        # Final layer norm
            self.ln_f = nn.LayerNorm(embed_dim)
        # Output head: project back to vocab logits
            self.lm_head = nn.Linear(embed_dim, vocab_size,bias = False)
        def forward(self,idx, targets=None):
            B,T = idx.shape
            # Embed tokens and positions
            x = self.embedding(idx)
            # Pass through all transformer blocks
            x = self.blocks(x)
            # Final layer norm
            x = self.ln_f(x) 
            # Project to logits
            logits = self.lm_head(x) 
            # Compute loss if targets provided
            loss = None
            
            if targets is not None:
                B,T,C = logits.shape
                loss = F.cross_entropy(
                    logits.view(B*T,C),
                    targets.view(B*T),
                    ignore_index = -1 # ignore padding if needed
                    )
            
            return logits, loss