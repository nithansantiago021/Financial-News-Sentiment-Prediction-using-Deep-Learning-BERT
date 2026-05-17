# Financial News Sentiment Prediction using Deep Learning & BERT

A production-ready sentiment classification system that labels finance-related tweets as **Bearish 🔴**, **Bullish 🟢**, or **Neutral 🔵** using deep learning. Implements three RNN-based baseline models (SimpleRNN, LSTM, GRU) and fine-tunes **FinBERT** (ProsusAI/finbert) for best-in-class performance. Shipped with an interactive Streamlit dashboard for live inference.

---

## Project Structure

```
financial-news-sentiment/
│
├── Financial_News_Sentiment_Prediction_Complete.ipynb   ← Main notebook (all sections)
│
├── app.py                 ← Streamlit dashboard (5 tabs)
├── export_vocab.py        ← Bridge script: notebook → dashboard
├── requirements.txt       ← Python dependencies
├── README.md              ← Dashboard-specific docs
├── models/                ← Created after running notebook
│   ├── vocab.json         ← Created by export_vocab.py
│   ├── SimpleRNN_best.pt
│   ├── LSTM_best.pt
│   └── GRU_best.pt
│
├── README.md                  
└── project_report.md      
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/nithansantiago021/Financial-News-Sentiment-Prediction-using-Deep-Learning-BERT.git
cd Financial-News-Sentiment-Prediction-using-Deep-Learning-BERT
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## How to Run

### Step 1 — Train all models (notebook)

Open and run every cell in the notebook from top to bottom:

```bash
jupyter notebook Financial_News_Sentiment_Prediction_Complete.ipynb
```

This will:
- Download the dataset from HuggingFace automatically
- Preprocess tweets with the full cleaning pipeline
- Train SimpleRNN, LSTM, and GRU with early stopping
- Fine-tune FinBERT (GPU recommended; pre-recorded results used on CPU)
- Save model weights to `models/`

### Step 2 — Export vocabulary

```bash
python export_vocab.py
```

This rebuilds the word→index mapping using the exact same pipeline as the notebook and saves it to `models/vocab.json`. The dashboard needs this file to convert new text to integers at inference time.

```bash
# Optional: verify vocab is consistent with saved model weights
python export_vocab.py --verify
```

### Step 3 — Launch the Streamlit dashboard

```bash
streamlit run app.py
```
---