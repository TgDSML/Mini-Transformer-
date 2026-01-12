# 🧠 Mini-Transformer (From Scratch)

This repository contains a **minimal yet complete implementation of a Transformer encoder built from scratch in PyTorch**.  
The project is designed as a **learning-oriented, research-style implementation**, focusing on understanding and validating the **core building blocks of modern Transformer architectures** rather than relying on high-level libraries.

The implementation closely follows the original paper:  
**“Attention Is All You Need” (Vaswani et al., 2017)**

---

## 📌 Project Objectives

- Implement a **Transformer encoder from scratch** using PyTorch
- Gain a **deep, practical understanding** of:
  - Self-attention
  - Multi-head attention
  - Positional encodings
  - Feed-forward networks
  - Residual connections & layer normalization
- Build a **clean, modular, and testable codebase**
- Validate correctness using **unit tests**
- Serve as a **reference project** for understanding Transformer internals

---

## 🧩 Core Components Implemented

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
  - Sinusoidal
  - Learnable (optional)
- Position-wise Feed-Forward Network
- Transformer Encoder Layer
- Residual connections & Layer Normalization
- End-to-end forward pass
- Gradient-safe architecture (verified via tests)

---

## 📂 Repository Structure

Mini-Transformer-/
│
├── data/
│   └── (optional datasets or toy data)
│
├── src/
│   ├── layers/
│   │   ├── attention.py
│   │   ├── feedforward.py
│   │   ├── positional_encoding.py
│   │   └── normalization.py
│   │
│   ├── models/
│   │   └── encoder.py
│   │
│   ├── utils/
│   │   └── helper functions
│   │
│   └── __init__.py
│
├── tests/
│   ├── test_attention.py
│   ├── test_feedforward.py
│   ├── test_positional_encoding.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md

---

## 🧪 Testing Philosophy

This project places strong emphasis on **correctness and reliability**.

Unit tests verify:

- Shape consistency
- Proper attention behavior
- Residual connections preserving dimensions
- Positional encoding correctness
- Gradient flow through the encoder

Run all tests with:

    pytest -q

---

## ⚙️ Installation & Setup

Clone repository:

    git clone https://github.com/TgDSML/Mini-Transformer-.git
    cd Mini-Transformer-

Create virtual environment:

    python -m venv .venv

Activate environment:

Windows:
    .\.venv\Scripts\Activate.ps1

macOS / Linux:
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

---

## 🚀 How to Run

Run the main script:

    python main.py

The `main.py` file performs a **forward pass through the Transformer encoder**, typically using randomly generated input or a small toy example, to validate:

- Model construction
- Forward propagation
- Output shapes
- Gradient flow

> This project is intentionally minimal and does **not** include full training on large datasets.

---

## 🧠 Design Philosophy

- **Clarity over abstraction**
- **Explicit implementations** instead of magic wrappers
- **Educational value first**
- Modular components that mirror the Transformer paper structure
- Suitable for:
  - Learning
  - Teaching
  - Interview preparation
  - Further research extensions

---

## 📈 Project Status

- ✅ Scaled Dot-Product Attention implemented
- ✅ Multi-Head Attention implemented
- ✅ Positional Encodings (sinusoidal & learnable)
- ✅ Transformer Encoder Layer assembled
- ✅ Comprehensive unit tests
- 🚧 Extensions & experimentation ongoing

---

## 🔮 Future Improvements

- Decoder implementation
- Full Transformer (Encoder-Decoder)
- Training loop on a toy language modeling task
- Attention visualization
- Benchmark against PyTorch’s `nn.Transformer`

---

## 📚 References

- Vaswani et al., *Attention Is All You Need*, 2017
- PyTorch documentation

---

## 📌 Notes

This repository is part of a **hands-on learning journey into Transformers and modern NLP architectures**, and is intentionally kept lightweight and readable.


