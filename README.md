# NeilGPT

> A GPT-style transformer built from scratch and trained on personal chat data to generate text in my conversational style.

## 🎯 Project Overview

This project implements a decoder-only transformer architecture (inspired by "Attention Is All You Need") from scratch using PyTorch. The model is trained on personal iMessage and Instagram chat data to learn and mimic my conversational patterns.

**Key Features:**
- ✅ Transformer architecture implemented from scratch (no `transformers` library)
- ✅ Privacy-first: Mandatory PII filtering and content sanitization
- ✅ End-to-end ML pipeline: data extraction → preprocessing → training → deployment
- ✅ Professional deployment: FastAPI backend + Hugging Face Space
- ✅ Portfolio-ready with clean architecture and documentation

## 🏗️ Architecture

- **Model**: Decoder-only GPT-style transformer
- **Framework**: PyTorch
- **Tokenization**: BPE (Byte-Pair Encoding)
- **Training**: AdamW optimizer with warmup + cosine decay
- **Deployment**: FastAPI + Hugging Face Spaces

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for NER (name detection)
python -m spacy download en_core_web_sm
```

### Data Extraction

**iMessage (macOS only):**
```bash
# Grant Full Disk Access to Terminal in System Preferences → Privacy & Security
python scripts/extract_data.py
```

**Instagram:**
1. Download your Instagram data (Settings → Privacy → Download Your Information)
2. Extract the ZIP file
3. Run: `python scripts/extract_data.py --instagram-path /path/to/instagram-export`

### Project Status

🚧 **In Development** - Currently implementing:
- [x] Data extraction pipeline (iMessage + Instagram)
- [x] Content filtering (PII removal, sensitive data)
- [x] Data preprocessing
- [ ] BPE Tokenization
- [ ] Transformer model implementation
- [ ] Training pipeline
- [ ] Text generation
- [ ] FastAPI backend
- [ ] Hugging Face Space deployment

## 📁 Project Structure

```
NeilGPT/
├── src/
│   ├── data/           # Data extraction & preprocessing
│   ├── tokenizer/      # BPE tokenization
│   ├── model/          # Transformer architecture
│   ├── training/       # Training loop
│   └── inference/      # Text generation
├── config/            # Configuration files
├── scripts/           # Entry point scripts
├── notebooks/         # Jupyter notebooks for experimentation
└── tests/             # Unit tests
```

## 🔒 Privacy & Safety

**This project takes privacy seriously:**
- All personal data is filtered before training (mandatory step)
- PII detection removes: phone numbers, emails, names, addresses, credit cards, SSNs
- Financial information is completely removed
- Sensitive topics (health, politics specifics) are filtered
- Data never leaves your machine unless you explicitly deploy

## 🎓 Learning Goals

This project demonstrates:
1. **Deep Learning**: Implementing transformers from first principles
2. **NLP**: Tokenization, sequence modeling, text generation
3. **Software Engineering**: Clean architecture, testing, documentation
4. **Data Engineering**: ETL pipelines, data cleaning, privacy filtering
5. **ML Engineering**: Training loops, checkpointing, evaluation
6. **Deployment**: API design, model serving, production deployment

## 📊 Results

*(Coming soon - training curves, sample generations, perplexity scores)*

## 🚀 Deployment

- **Website Integration**: FastAPI backend integrated with personal Vercel site
- **Public Demo**: Hugging Face Space for interactive testing
- **API**: RESTful API for programmatic access

## 🙏 Acknowledgments

- "Attention Is All You Need" (Vaswani et al., 2017)
- PyTorch team
- Hugging Face community

## 📝 License

MIT License - See LICENSE file for details

---

**Built by Neil Tripathi** | [Website](https://your-website.com) | [LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/neilt93)
