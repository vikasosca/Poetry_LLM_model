import tiktoken

class CharTokenizer:
    def __init__(self,text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        #Mappings
        self.stoi = {ch: i for i,ch in (enumerate(self.chars))}
        self.itos = {i: ch for i,ch in (enumerate(self.chars))}
        
    def encode(self, s: str):
         """Convert string to list of integers"""
         return [self.stoi[c] for c in s]
    def decode(self, I: int):
        return ''.join([self.itos[i] for i in I]  )

class BPETokenizer:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")
    def encode(self,text):
        return self.encoder.encode(text)
    def decode(self,ids):
        return self.encoder.decode(ids)
    @property
    def vocab_size(self):
        return self.encoder.n_vocab

    

        