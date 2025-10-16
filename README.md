# 🧠 RAG-Flow-Backend

This repository contains the backend implementation for a RAG (Retrieval-Augmented Generation) application, likely utilizing models like **PyTorch** and vector databases like **ChromaDB**.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed on your system:

* **Python 3.9+** (Check with: `python --version`)
* **Git** (Check with: `git --version`)

---

## 💻 Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository

Clone the project from GitHub using the terminal:

```bash
git clone [https://github.com/PrayagN/RAG-Flow-Backend.git](https://github.com/PrayagN/RAG-Flow-Backend.git)
cd RAG-Flow-Backend

  python -m venv venv
  venv\Scripts\activate
  uvicorn app.main:app --reload
