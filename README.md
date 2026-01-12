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
- Positional Encoding (sinusoidal & learnable)
- Position-wise Feed-Forward Network
- Transformer Encoder Layer
- Residual connections & Layer Normalization
- End-to-end forward pass
- Gradient-safe architecture (verified via tests)

---

## 📂 Repository Structure

- data/
- src/
  - layers/
    - attention.py
    - feedforward.py
    - positional_encoding.py
    - normalization.py
  - models/
    - encoder.py
  - utils/
  - __init__.py
- tests/
  - test_attention.py
  - test_feedforward.py
  - test_positional_encoding.py
- main.py
- requirements.txt
- .gitignore
- README.md

---

## 🧪 Testing

Run all unit tests:

    pytest -q

---

## ⚙️ Installation & Setup

Clone repository:

    git clone https://github.com/TgDSML/Mini-Transformer-.git
    cd Mini-Transformer-

Create virtual environment:

    python -m venv .venv

Activate:

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

The script performs a forward pass through the Transformer encoder using toy or random input to validate correctness and gradient flow.

---

## 🧠 Design Philosophy

- Clarity over abstraction
- Explicit implementations
- Educational focus
- Modular, testable components
- Suitable for learning, teaching, interviews, and research

---

## 📈 Project Status

- ✅ Core Transformer components implemented
- ✅ Encoder assembled
- ✅ Unit-tested
- 🚧 Extensions ongoing

---

## 🔮 Future Improvements

- Decoder & full Transformer
- Training loop
- Attention visualization
- Benchmark vs PyTorch Transformer

---

## 📚 References

- Vaswani et al., *Attention Is All You Need*, 2017
- PyTorch documentation



