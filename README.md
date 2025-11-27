# Poetry_LLM_model

Train model for poetry from scratch and test it.
🌟 Overview
This project demonstrates how to build a fully functional language model from scratch using pure PyTorch — no Hugging Face shortcuts, no pre-trained weights. The model blends classical English poetry with Indian epic storytelling to create original, cross-cultural verses.

"The moon rose over the forest, where fairies danced among daffodils." 

🚀 Features
✅ Built from scratch: Custom Transformer implementation with attention, LayerNorm, and GELU
✅ Multilingual poetry: Trained on Shakespeare, Dickinson, Wordsworth, and Ramayana
✅ Efficient architecture: 50M parameters, 768-dim embeddings, 512-token context
✅ Production-ready API: FastAPI with health checks, rate limiting, and async support
✅ CPU training: Fully trainable on consumer hardware
✅ BPE tokenization: Uses GPT-2's 50K vocabulary for rich linguistic coverage

🛠️ Technical Architecture
Model Specifications
Architecture: Custom Transformer (6 blocks)
Embedding Dimension: 768
Context Length: 512 tokens
Vocabulary Size: 50,257 (GPT-2 BPE)
Parameters: ~50 million
Activation: GELU
Normalization: LayerNorm (post-norm)
Attention: Causal self-attention with learned positional embeddings
Training Details
Framework: PyTorch 2.0+
Optimizer: AdamW with gradient clipping
Training Steps: 55,000
Hardware: CPU (fully trainable on laptop)
Memory Management: Dynamic cleanup with gc.collect()
Batch Size: 16-32 (CPU-optimized)
🚀 Quick Start
Prerequisites
Python 3.8+
PyTorch 2.0+
8GB+ RAM (16GB recommended)
Installation
bash
git clone https://github.com/your-username/ai-poetry-generator.git
cd ai-poetry-generator
pip install -r requirements.txt

git clone https://github.com/your-username/ai-poetry-generator.git
cd ai-poetry-generator
pip install -r requirements.txt

Download Trained Model
Download the pre-trained model checkpoint:

# Download from releases (or use your own)
wget https://github.com/your-username/ai-poetry-generator/releases/download/v1.0/tinyllm_checkpoint.pth

Test Generation
python test_poetry_model.py

Run API Server
uvicorn poetry_api:app --host 0.0.0.0 --port 8000
Visit http://localhost:8000/docs for interactive API documentation.

🌐 API Endpoints
Generate Poetry
POST /generate

Request:
{
  "prompt": "The moon rises",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_k": 50
}

Response:
{
  "generated": "The moon rises over the forest, where Hanuman danced among daffodils...",
  "prompt": "The moon rises",
  "max_tokens": 100,
  "temperature": 0.7
}

Health Check
http: GET /health
Response:
{
  "status": "healthy",
  "model_loaded": true
}
📦 Project Structure

ai-poetry-generator/
├── model.py              # Custom Transformer implementation
├── poetry_api.py         # FastAPI application
├── test_poetry_model.py  # CLI testing script
├── train.py              # Training script
├── requirements.txt      # Dependencies
├── tinyllm_checkpoint.pth # Trained model (500MB)
├── input.txt             # Training corpus (multilingual poetry)
└── README.md
ai-poetry-generator/
├── model.py              # Custom Transformer implementation
├── poetry_api.py         # FastAPI application
├── test_poetry_model.py  # CLI testing script
├── train.py              # Training script
├── requirements.txt      # Dependencies
├── tinyllm_checkpoint.pth # Trained model (500MB)
├── input.txt             # Training corpus (multilingual poetry)
└── README.md

🧪 Training Your Own Model
Prepare Training Data
Collect public domain poetry in input.txt
Ensure UTF-8 encoding with consistent formatting
Balance corpus across languages/styles
Start Training

python train.py
Training Configuration
Embed Dim: 768
Block Size: 512
Num Blocks: 6
Learning Rate: 3e-4 (with decay)
Batch Size: 16-32 (CPU optimized)
Steps: 55,000
Memory Management
For CPU training, the script includes automatic memory cleanup every 1000 steps to prevent OOM errors.

🎨 Sample Outputs
Prompt: "The moon rises"

"The moon rises over the forest, where fairies danced among daffodils, and Rama's bow gleamed like Wordsworth's clouds." 

Prompt: "To be or not to be"

"To be or not to be: that is the question of dharma, where exile meets devotion in the forest of Dandaka." 

Prompt: "In the forest of my dreams"

"In the forest of my dreams, where daffodils bloom beside the sacred, Tis the river of memories fllowing silent." 

🤝 Contributing
Contributions are welcome! Please open an issue or submit a PR for:

Additional poetry corpora
Improved training techniques
Web interface enhancements
Multilingual extensions
