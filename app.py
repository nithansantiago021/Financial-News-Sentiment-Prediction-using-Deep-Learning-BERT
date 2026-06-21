# ── Imports ───────────────────────────────────────────────────────────────────
import re
import os
import json
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
import nltk

for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

try:
    from nltk.corpus import stopwords as _sw_corpus
    from nltk.stem import WordNetLemmatizer
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

from sklearn.metrics import confusion_matrix   # noqa: F401  (used later)

# ── Transformers (FinBERT) ────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download  # noqa: F401  (used later)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSentiment AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4 { font-family:'IBM Plex Mono',monospace; letter-spacing:-0.4px; }

.stApp { background:#07101f; }
.main .block-container { padding-top:1.4rem; padding-bottom:2rem; max-width:1280px; }

section[data-testid="stSidebar"] { background:#050c18; border-right:1px solid #152030; }
section[data-testid="stSidebar"] * { color:#b8cce8 !important; }

.stTabs [data-baseweb="tab-list"] { gap:3px; border-bottom:1px solid #152030; }
.stTabs [data-baseweb="tab"] {
    font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
    padding:8px 20px; border-radius:6px 6px 0 0;
    background:#0a1525; border:1px solid #152030; border-bottom:none; color:#5a7a9a;
}
.stTabs [aria-selected="true"] { background:#0f2040 !important; color:#88bbee !important; border-color:#2a4060 !important; }

div.stButton > button {
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
    border-radius:6px; border:1px solid #1e3050;
    background:#0a1828; color:#88aacc; transition:all 0.15s;
}
div.stButton > button:hover { background:#132538; border-color:#3498db; color:#c8e0f8; }
div.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#1a3a6a 0%,#0d2244 100%);
    border-color:#3a6aaa; color:#c8e0ff; font-weight:600;
}

.stTextArea textarea {
    background:#0a1525 !important; border:1px solid #1e3050 !important;
    color:#c8d8f0 !important; font-size:0.9rem !important; border-radius:8px !important;
}

.dash-title { font-family:'IBM Plex Mono',monospace; font-size:1.75rem; font-weight:700; color:#ddeeff; letter-spacing:-1px; margin-bottom:2px; }
.dash-sub   { font-size:0.83rem; color:#4a6888; margin-bottom:1.4rem; }

.metric-tile { background:#0a1525; border:1px solid #1e3050; border-radius:8px; padding:14px 18px; text-align:center; }
.metric-tile .val { font-size:1.7rem; font-weight:700; font-family:'IBM Plex Mono',monospace; line-height:1.1; }
.metric-tile .lbl { font-size:0.68rem; color:#4a6888; text-transform:uppercase; letter-spacing:1.2px; margin-top:4px; }

.pred-card  { border-radius:10px; padding:20px 26px; margin:10px 0; font-family:'IBM Plex Mono',monospace; }
.pred-bearish { background:linear-gradient(135deg,#1a0608 0%,#2a0c10 100%); border-left:4px solid #e74c3c; }
.pred-bullish { background:linear-gradient(135deg,#021408 0%,#0a2414 100%); border-left:4px solid #2ecc71; }
.pred-neutral { background:linear-gradient(135deg,#020c1a 0%,#081828 100%); border-left:4px solid #3498db; }
.pred-label { font-size:1.5rem; font-weight:700; margin-bottom:3px; }
.pred-conf  { font-size:0.82rem; color:#8899aa; }

.prob-row   { display:flex; align-items:center; margin-bottom:9px; gap:10px; }
.prob-label { width:68px; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#88aabb; }
.prob-bar-wrap { flex:1; background:#0e1e30; border-radius:3px; height:16px; overflow:hidden; }
.prob-bar   { height:100%; border-radius:3px; }
.prob-pct   { width:46px; text-align:right; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:#7090a0; }

.sent-block { background:#080f1c; border-radius:0 7px 7px 0; padding:9px 14px; margin-bottom:7px; font-size:0.82rem; }
.insight-box { background:#080f1c; border:1px solid #1a3050; border-radius:7px; padding:10px 14px; font-size:0.8rem; color:#4a6888; line-height:1.65; }
.badge-ok   { background:#071a0e; border:1px solid #1a4428; border-radius:7px; padding:8px 12px; color:#3a8a4a; font-size:0.76rem; }
.badge-demo { background:#0f1420; border:1px solid #2a3560; border-radius:7px; padding:8px 12px; color:#5a7aaa; font-size:0.76rem; }
.badge-warn { background:#1a0a08; border:1px solid #4a2018; border-radius:7px; padding:8px 12px; color:#aa5040; font-size:0.76rem; }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

SEED        = 42
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
NUM_CLASSES = 3
MAX_SEQ_LEN = 32
PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
LABEL_MAP   = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
LABEL_ORDER = ["Bearish", "Bullish", "Neutral"]
COLORS      = {"Bearish": "#e74c3c", "Bullish": "#2ecc71", "Neutral": "#3498db"}
EMOJIS      = {"Bearish": "🔴", "Bullish": "🟢", "Neutral": "🔵"}
CARD_CLS    = {"Bearish": "pred-bearish", "Bullish": "pred-bullish", "Neutral": "pred-neutral"}

torch.manual_seed(SEED)
np.random.seed(SEED)
# Auto-detect: use the GPU (RTX 5060 / any CUDA device) when available,
# fall back to CPU automatically so the app never crashes on a CPU-only machine.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPOSITORY_ID = "ntini97/financial-sentiment-models"

# ═════════════════════════════════════════════════════════════════════════════
# MODEL CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.rnn(self.dropout(self.embedding(x)))
        return self.fc(self.dropout(h[-1]))


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(self.dropout(self.embedding(x)))
        return self.fc(self.dropout(h[-1]))


class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, h = self.gru(self.dropout(self.embedding(x)))
        return self.fc(self.dropout(h[-1]))


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _build_nlp():
    if not _NLTK_OK:
        return set(), None
    stop = set(_sw_corpus.words("english"))
    keep = {"up","down","not","no","nor","against","below","above","under","over"}
    return stop - keep, WordNetLemmatizer()

_STOP_WORDS, _LEM = _build_nlp()


def clean_tweet_rnn(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+",  "",    text)
    text = re.sub(r"@\w+",            "",    text)
    text = re.sub(r"\$([A-Za-z]+)",   r"\1", text)
    text = re.sub(r"#(\w+)",          r"\1", text)
    text = re.sub(r"[^a-z\s]",        "",    text)
    text = re.sub(r"\s+",             " ",   text).strip()
    if _LEM is None:
        return text
    return " ".join(_LEM.lemmatize(t) for t in text.split()
                    if t not in _STOP_WORDS and len(t) > 2)


def light_clean_bert(text: str) -> str:
    """Minimal cleaning for FinBERT — only strip URLs and normalise whitespace.
    FinBERT's sub-word tokeniser handles punctuation, casing and $tickers natively.
    """
    text = re.sub(r"http\S+|www\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def text_to_sequence(text: str, word2idx: dict) -> list:
    unk = word2idx.get(UNK_TOKEN, 1)
    pad = word2idx.get(PAD_TOKEN, 0)
    seq = [word2idx.get(t, unk) for t in text.split()][:MAX_SEQ_LEN]
    return seq + [pad] * (MAX_SEQ_LEN - len(seq))


# ═════════════════════════════════════════════════════════════════════════════
# MODEL / VOCAB LOADING
# ═════════════════════════════════════════════════════════════════════════════

# MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@st.cache_resource(show_spinner=False)
def load_vocab():
    try:
        # Dynamically download vocab.json from the weights/ directory in your cloud hub
        cached_vocab_path = hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename="weights/vocab.json"
        )
        with open(cached_vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading vocabulary from cloud: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_model(name: str, vocab_size: int):
    cls_map = {"SimpleRNN": SentimentRNN, "LSTM": SentimentLSTM, "GRU": SentimentGRU}
    if name not in cls_map:
        return None, f"Unknown model: {name}"
    
    try:
        # 1. Dynamically download the target weights file from your cloud Model Hub
        # This looks into your HF repo's "weights/" folder for files like "SimpleRNN_best.pt"
        cached_weight_path = hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename=f"weights/{name}_best.pt"
        )
        
        # 2. Instantiate the corresponding architecture blueprint
        m = cls_map[name](vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS, DROPOUT, 0)
        
        # 3. Load the weights using the dynamic cloud path and apply our CPU failsafe
        m.load_state_dict(torch.load(cached_weight_path, map_location=device))
        m.to(device)
        m.eval()
        return m, None
        
    except Exception as exc:
        return None, f"Failed to load model from cloud hub: {str(exc)}"


@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_finbert():
    if not _TRANSFORMERS_OK:
        return None, None, "transformers library not installed (pip install transformers)"

    # The official base model name where tokenizer configurations live
    BASE_MODEL = "ProsusAI/finbert"
    REPOSITORY_ID = "ntini97/financial-sentiment-models"

    try:
        # 1. Fetch the official tokenizer from the main FinBERT repo
        tokeniser = AutoTokenizer.from_pretrained(BASE_MODEL)
        
        # 2. Instantiate the base model structure from the main FinBERT repo
        bert = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        
        # 3. Dynamically download JUST your custom fine-tuned weights file from YOUR repository
        # Adjust the filename path if your file is named differently in your HF model tree
        cached_weight_path = hf_hub_download(
            repo_id=REPOSITORY_ID,
            filename="weights/FinBERT_best.pt" 
        )
        
        # 4. Load your custom fine-tuned states into the model structure
        state = torch.load(cached_weight_path, map_location=device)
        bert.load_state_dict(state)
        
        bert.to(device)
        bert.eval()
        return tokeniser, bert, None
    except Exception as exc:
        return None, None, str(exc)

try:
    # Dynamically download the training logs/metrics file from the cloud repo
    cached_metrics_path = hf_hub_download(
        repo_id=REPOSITORY_ID,
        filename="weights/model_training_results.json"
    )
    with open(cached_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
except Exception as e:
    st.error(f"Error loading metrics from cloud: {e}")
    metrics = {}  # Fallback to an empty dictionary to prevent the UI from crashing
# ═════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═════════════════════════════════════════════════════════════════════════════
def predict_single(text: str, model, word2idx: dict) -> dict:
    cleaned = clean_tweet_rnn(text) or "neutral"
    tensor  = torch.tensor([text_to_sequence(cleaned, word2idx)], dtype=torch.long).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return {"label": LABEL_MAP[idx], "confidence": float(probs[idx]),
            "probs": {LABEL_MAP[i]: float(probs[i]) for i in range(3)}, "cleaned": cleaned}


def predict_paragraph(text: str, model, word2idx: dict) -> dict:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) > 3]
    if not sents:
        sents = [text]
    results = [predict_single(s, model, word2idx) for s in sents]
    votes   = Counter(r["label"] for r in results)
    return {"overall": votes.most_common(1)[0][0], "votes": dict(votes),
            "sentences": [{"text": s, **r} for s, r in zip(sents, results)]}


def demo_predict(text: str) -> dict:
    """Deterministic simulated output when no model weights are available."""
    rng   = np.random.RandomState(abs(hash(text)) % (2**31))
    probs = rng.dirichlet([0.6, 1.0, 3.5])
    idx   = int(np.argmax(probs))
    return {"label": LABEL_MAP[idx], "confidence": float(probs[idx]),
            "probs": {LABEL_MAP[i]: float(probs[i]) for i in range(3)},
            "cleaned": clean_tweet_rnn(text)}


# ── FinBERT-specific inference ────────────────────────────────────────────────
MAX_BERT_LEN = 64   # max sub-word tokens; matches notebook training setting

def predict_single_bert(text: str, tokeniser, bert_model) -> dict:
    """Run one raw tweet through the fine-tuned FinBERT model.

    Uses light_clean_bert (URL removal only) — FinBERT's sub-word tokeniser
    handles punctuation, casing, and $tickers natively. Tensors are moved to
    the same device (GPU if available) that the model was loaded onto.
    """
    cleaned = light_clean_bert(text)
    enc = tokeniser(
        cleaned,
        max_length     = MAX_BERT_LEN,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(
            input_ids      = enc["input_ids"],
            attention_mask = enc["attention_mask"],
        ).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "label":     LABEL_MAP[idx],
        "confidence": float(probs[idx]),
        "probs":     {LABEL_MAP[i]: float(probs[i]) for i in range(3)},
        "cleaned":   cleaned,
    }


def predict_paragraph_bert(text: str, tokeniser, bert_model) -> dict:
    """Sentence-level FinBERT inference with majority-vote aggregation."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
             if len(s.split()) > 3]
    if not sents:
        sents = [text]
    results = [predict_single_bert(s, tokeniser, bert_model) for s in sents]
    votes   = Counter(r["label"] for r in results)
    return {"overall": votes.most_common(1)[0][0], "votes": dict(votes),
            "sentences": [{"text": s, **r} for s, r in zip(sents, results)]}


# ═════════════════════════════════════════════════════════════════════════════
# STATIC DATA
# ═════════════════════════════════════════════════════════════════════════════

TRAIN_DIST = {"Bearish": 1385, "Bullish": 1947, "Neutral": 6606}
VAL_DIST   = {"Bearish": 346,  "Bullish": 484,  "Neutral": 1656}

TOP_WORDS = {
    "Bearish": [("stock",612),("market",581),("decline",443),("fall",398),("price",371),
                ("loss",355),("sell",318),("drop",307),("weak",289),("concern",265),
                ("cut",248),("miss",231),("down",219),("warning",208),("risk",197)],
    "Bullish": [("stock",744),("market",703),("gain",588),("rise",521),("growth",487),
                ("revenue",463),("buy",441),("beat",418),("strong",392),("record",371),
                ("profit",348),("surge",325),("high",299),("up",278),("upgrade",255)],
    "Neutral": [("stock",2103),("market",1982),("company",1654),("share",1512),("report",1387),
                ("quarter",1205),("revenue",1089),("billion",976),("year",932),("analyst",889),
                ("rate",812),("new",754),("said",701),("per",678),("price",645)],
}
TOP_TICKERS = {
    "Bearish": [("$AAPL",88),("$TSLA",76),("$AMZN",71),("$META",64),("$NVDA",58),
                ("$GOOGL",52),("$MSFT",49),("$GS",44),("$BAC",39),("$F",34)],
    "Bullish": [("$TSLA",142),("$NVDA",128),("$AAPL",115),("$AMZN",99),("$MSFT",94),
                ("$META",87),("$AMD",72),("$GOOGL",61),("$NFLX",55),("$CRM",48)],
    "Neutral": [("$AAPL",312),("$MSFT",278),("$AMZN",241),("$TSLA",219),("$GOOGL",198),
                ("$META",187),("$NVDA",171),("$JPM",145),("$BAC",132),("$GS",118)],
}
SAMPLES = {
    "🔴 Bearish": [
        "$AAPL stock drops 12% after weak iPhone sales guidance — analysts warn of further downside",
        "Fed signals prolonged rate hikes; equity markets face significant headwinds this quarter",
        "$AMZN Q4 earnings missed expectations. Revenue growth decelerated; cautious guidance issued.",
        "Recession fears mount as unemployment rises and consumer confidence hits 18-month low",
    ],
    "🟢 Bullish": [
        "$TSLA crushed Q3 earnings — revenue up 25% YoY, raising full-year guidance",
        "$NVDA reports record revenue driven by surging AI chip demand; shares up 15% after-hours",
        "Strong jobs report fuels optimism; markets rally across all major indices today",
        "$META ad revenue beats consensus by 18%; buyback programme doubled to $50 billion",
    ],
    "🔵 Neutral": [
        "Federal Reserve holds interest rates steady pending further economic data",
        "$MSFT announces partnership with OpenAI; financial terms were not disclosed",
        "S&P 500 ends flat as investors weigh mixed earnings results against macro uncertainty",
        "Central bank minutes reveal divided views on the pace of future rate changes",
    ],
}

# ═════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _dark_fig(nrows=1, ncols=1, figsize=(8, 4), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor("#07101f")
    ax_iter = np.array(axes).flat if hasattr(axes, "__iter__") else [axes]
    for ax in ax_iter:
        ax.set_facecolor("#0a1525")
        ax.tick_params(colors="#4a6888", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#152030")
    return fig, axes


def _render(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _style_ax(ax):
    ax.set_facecolor("#0a1525")
    ax.tick_params(colors="#4a6888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#152030")


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-family:IBM Plex Mono,monospace;font-size:1.05rem;"
        "font-weight:700;color:#ddeeff;margin-bottom:1px;'>FinSentiment AI</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='font-size:0.68rem;color:#3a5878;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:5px;'>Active Model</div>", unsafe_allow_html=True)
    selected_model = st.selectbox("Active Model", ["LSTM","GRU","Simple RNN","FinBERT"], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#152030;margin:0.9rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#3a5878;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;'>Compute Device</div>", unsafe_allow_html=True)
    
    # Dynamic hardware display works perfectly on Windows GPU and HF CPU spaces!
    if device.type == "cuda":
        _gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f"<div class='badge-ok'>✓ GPU — {_gpu_name}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-warn'>⚠ Running on CPU — no CUDA GPU detected</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#152030;margin:0.9rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#3a5878;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px;'>Dataset</div>", unsafe_allow_html=True)
    
    # Fixed hardcoded layout details display beautifully
    for _lbl, _val in [("Source","HuggingFace zeroshot/…"),("Train","9,938 tweets"),
                        ("Val","2,486 tweets"),("Classes","Bearish · Bullish · Neutral"),("Vocab","~6,308 tokens")]:
        st.markdown(f"<div style='font-size:0.75rem;color:#6a8aaa;margin-bottom:3px;'><span style='color:#2a4060;'>{_lbl}:</span> {_val}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#152030;margin:0.9rem 0;'>", unsafe_allow_html=True)

    # ── Resolve which model to load ──────────────────────────────────────────
    _is_bert  = selected_model == "FinBERT"
    _mk_map   = {"LSTM": "LSTM", "GRU": "GRU", "Simple RNN": "SimpleRNN"}
    _mk       = _mk_map.get(selected_model, "LSTM")

    # Cloud network fetches execution blocks
    vocab          = load_vocab()
    active_model   = None       
    word2idx       = vocab or {}
    model_ok       = False      
    model_err      = ""

    bert_tokeniser = None
    bert_model_obj = None
    bert_ok        = False
    bert_err       = ""

    if _is_bert:
        bert_tokeniser, bert_model_obj, bert_err = load_finbert()
        bert_ok = bert_model_obj is not None
    elif vocab:
        active_model, model_err = load_model(_mk, len(vocab))
        model_ok = active_model is not None

    # ── UI Status Indicators (Updated to reflect Cloud Repository Paths) ─────
    if _is_bert and bert_ok:
        st.markdown(
            "<div class='badge-ok'>✓ FinBERT loaded <br></div>",
            unsafe_allow_html=True,
        )
    elif _is_bert and not bert_ok:
        st.markdown(
            f"<div class='badge-warn'>⚠ FinBERT Load Error<br>"
            f"<span style='color:#6a3828;font-size:0.68rem;font-family:IBM Plex Mono,monospace;'>"
            f"{bert_err or 'Check Hugging Face Model Hub repo paths'}</span></div>",
            unsafe_allow_html=True,
        )
    elif model_ok:
        st.markdown(f"<div class='badge-ok'>✓ {selected_model} loaded </div>", unsafe_allow_html=True)
    else:
        # Informative cloud connection validation string display layout
        _fail_reason = model_err if model_err else "Could not resolve vocab.json from the Model Hub."
        st.markdown(
            f"<div class='badge-warn'>⚠ Cloud Load Error:<br>"
            f"<span style='color:#6a3828;font-size:0.68rem;font-family:IBM Plex Mono,monospace;'>"
            f"{_fail_reason}</span><br>"
            f"<span style='color:#4a2818;font-size:0.65rem;'>Verify repository permissions and names</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#152030;margin:0.9rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.65rem;color:#1e3048;line-height:1.7;'>Dataset: Twitter Financial News Sentiment<br>zeroshot/twitter-financial-news-sentiment<br></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# HEADER + TOP METRICS BAR
# ═════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<div class='dash-title'>Financial News Sentiment AI</div>"
    "<div class='dash-sub'>"
    "Classify finance tweets as Bearish · Bullish · Neutral "
    "using RNN baselines &amp; FinBERT"
    "</div>",
    unsafe_allow_html=True
)

# Select correct metrics key — map the sidebar label to the metrics dict key
# (e.g. "Simple RNN" -> "SimpleRNN") and use it for every model, not just FinBERT.
_met_key = "FinBERT" if _is_bert else _mk

# Get metrics for ONLY that model
_met = metrics[_met_key]
_cols = st.columns(5)

for _col, _lbl, _val, _clr in zip(
    _cols,
    ["Accuracy","Macro F1","Bearish F1","Bullish F1","Neutral F1"],
    [
        _met["Accuracy"],
        _met["Macro F1"],
        _met["Bearish F1"],
        _met["Bullish F1"],
        _met["Neutral F1"]
    ],
    ["#c8dcf8","#f0c040","#e74c3c","#2ecc71","#3498db"]
):

    _col.markdown(
        f"""
        <div class='metric-tile'>
            <div class='val' style='color:{_clr};'>
                {_val:.4f}
            </div>
            <div class='lbl'>
                {_lbl}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    "<div style='margin-bottom:1.2rem;'></div>",
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "Live Prediction", "Model Comparison",
    "Dataset Explorer", "Batch Analyse"
])

# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE RENDERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_prob_bars(probs: dict):
    for cls in LABEL_ORDER:
        p, c = probs[cls], COLORS[cls]
        st.markdown(
            f"<div class='prob-row'>"
            f"<span class='prob-label' style='color:{c};'>{cls}</span>"
            f"<div class='prob-bar-wrap'><div class='prob-bar' style='width:{int(p*100)}%;background:{c};'></div></div>"
            f"<span class='prob-pct'>{p:.1%}</span></div>",
            unsafe_allow_html=True,
        )


def render_pred_card(label: str, conf: float, model_name: str, cleaned: str = ""):
    st.markdown(
        f"<div class='pred-card {CARD_CLS[label]}'>"
        f"<div class='pred-label'>{EMOJIS[label]} {label}</div>"
        f"<div class='pred-conf'>Confidence: {conf:.1%} · Model: {model_name}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if cleaned:
        st.markdown(
            f"<div style='font-size:0.72rem;color:#2a4060;font-family:IBM Plex Mono,monospace;margin-top:4px;'>"
            f"Cleaned: <span style='color:#4a6888;'>{cleaned[:90]}{'…' if len(cleaned)>90 else ''}</span></div>",
            unsafe_allow_html=True,
        )


def run_inference(text: str) -> dict:
    """Route to FinBERT, RNN model, or demo depending on what is loaded."""
    if _is_bert and bert_ok:
        return predict_single_bert(text, bert_tokeniser, bert_model_obj)
    if model_ok and not _is_bert:
        return predict_single(text, active_model, word2idx)
    return demo_predict(text)


def run_para_inference(text: str) -> dict:
    """Paragraph-level routing: FinBERT, RNN, or demo."""
    if _is_bert and bert_ok:
        return predict_paragraph_bert(text, bert_tokeniser, bert_model_obj)
    if model_ok and not _is_bert:
        return predict_paragraph(text, active_model, word2idx)
    # Demo fallback — simulate sentence-level results
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip())
             if len(s.split()) > 3] or [text]
    results = [demo_predict(s) for s in sents]
    votes   = Counter(r["label"] for r in results)
    return {"overall": votes.most_common(1)[0][0], "votes": dict(votes),
            "sentences": [{"text": s, **r} for s, r in zip(sents, results)]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — LIVE PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    st.markdown("### Live Sentiment Prediction")
    st.markdown("<div class='insight-box'>Type any financial tweet, headline, or paragraph. Click <b>Predict Sentiment</b> for a single result or <b>Analyse Paragraph</b> for sentence-by-sentence breakdown with majority vote.</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

    # Sample buttons
    st.markdown("<div style='font-size:0.68rem;color:#3a5878;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;'>Quick samples — click to load</div>", unsafe_allow_html=True)
    _sc = st.columns(3)
    _clicked = None
    for _i, (_slbl, _hlines) in enumerate(SAMPLES.items()):
        _sc[_i].markdown(f"<div style='font-size:0.7rem;color:{list(COLORS.values())[_i]};text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;'>{_slbl}</div>", unsafe_allow_html=True)
        for _h in _hlines:
            if _sc[_i].button((_h[:52]+"…") if len(_h)>52 else _h, key=f"s{hash(_h)}"):
                _clicked = _h

    user_text = st.text_area("FINANCIAL TEXT", value=_clicked or "",
                             height=100, placeholder="e.g. $TSLA beats Q4 earnings, raises full-year guidance…")

    _bl, _br, _ = st.columns([1,1,2])
    _do_pred  = _bl.button("Predict Sentiment",  type="primary", use_container_width=True)
    _do_para  = _br.button("Analyse Paragraph",               use_container_width=True)

    if _is_bert and not bert_ok:
        st.info(f"ℹ FinBERT weights not found — running in demo mode. Place `FinBERT_best.pt` in `models/`. Error: {bert_err}")
    elif not model_ok and not _is_bert:
        st.info("ℹ Running in **demo mode** — place `vocab.json` and `*_best.pt` files in the `models/` folder to enable live inference.")

    # ── Single prediction ──────────────────────────────────────────────────────
    if _do_pred:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            st.markdown("---")
            res = run_inference(user_text)
            _l, _r = st.columns(2)
            with _l:
                render_pred_card(res["label"], res["confidence"], selected_model, res["cleaned"])
            with _r:
                st.markdown("<div style='font-size:0.72rem;color:#3a5878;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Class Probabilities</div>", unsafe_allow_html=True)
                render_prob_bars(res["probs"])

            # Bar chart
            fig, ax = _dark_fig(figsize=(7, 2.8))
            vals = [res["probs"][l] for l in LABEL_ORDER]
            bars = ax.bar(LABEL_ORDER, vals, color=[COLORS[l] for l in LABEL_ORDER], edgecolor="#152030", linewidth=0.8)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{v:.1%}",
                        ha="center", va="bottom", color="white", fontsize=9,
                        fontweight="bold", fontfamily="monospace")
            ax.set_ylim(0, 1.12)
            ax.set_title(f"Prediction: {EMOJIS[res['label']]} {res['label']}  ({res['confidence']:.1%} confidence)",
                         color="#c8d8f0", fontsize=9, fontfamily="monospace", pad=8)
            ax.set_ylabel("Probability", color="#4a6888", fontsize=8)
            _style_ax(ax)
            plt.tight_layout()
            _render(fig)

    # ── Paragraph analysis ─────────────────────────────────────────────────────
    if _do_para:
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            st.markdown("---")
            st.markdown("#### Paragraph Sentiment Breakdown")
            para = run_para_inference(user_text)
            overall = para["overall"]
            votes   = para["votes"]
            vote_html = " · ".join(f"<span style='color:{COLORS[k]};'>{EMOJIS[k]} {k}: {v}</span>" for k, v in votes.items())
            st.markdown(
                f"<div class='pred-card {CARD_CLS[overall]}'>"
                f"<div class='pred-label'>{EMOJIS[overall]} {overall}</div>"
                f"<div class='pred-conf'>Overall sentiment · majority vote from {sum(votes.values())} sentence(s)</div>"
                f"</div>"
                f"<div style='font-size:0.8rem;margin-bottom:1rem;'>{vote_html}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='font-size:0.72rem;color:#3a5878;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;'>Sentence Breakdown</div>", unsafe_allow_html=True)
            for sr in para["sentences"]:
                lbl = sr["label"]
                st.markdown(
                    f"<div class='sent-block' style='border-left:3px solid {COLORS[lbl]};'>"
                    f"<span style='color:{COLORS[lbl]};font-family:IBM Plex Mono,monospace;font-weight:600;font-size:0.78rem;'>"
                    f"{EMOJIS[lbl]} {lbl} ({sr['confidence']:.0%})</span><br>"
                    f"<span style='color:#8899bb;'>{sr['text']}</span></div>",
                    unsafe_allow_html=True,
                )
            # Sentence confidence chart (if >1 sentence)
            if len(para["sentences"]) > 1:
                ns  = len(para["sentences"])
                fig, ax = _dark_fig(figsize=(min(11, ns*2.5+1), 3))
                xs = np.arange(ns)
                for ci, cls in enumerate(LABEL_ORDER):
                    ax.bar(xs + ci*0.27, [sr["probs"][cls] for sr in para["sentences"]],
                           0.27, label=cls, color=COLORS[cls], edgecolor="#152030", linewidth=0.5)
                ax.set_xticks(xs+0.27)
                ax.set_xticklabels([f"S{i+1}" for i in xs], color="#4a6888", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Probability", color="#4a6888", fontsize=8)
                ax.legend(fontsize=7, facecolor="#07101f", labelcolor="#8899bb", edgecolor="#152030")
                _style_ax(ax)
                plt.tight_layout()
                _render(fig)

    # ── Static example (shown when nothing submitted yet) ──────────────────────
    if not _do_pred and not _do_para:
        st.markdown("---")
        st.markdown("<div style='font-size:0.75rem;color:#3a5878;margin-bottom:0.5rem;'>Example outputs for three contrasting headlines</div>", unsafe_allow_html=True)
        examples = [
            {"text": "$AAPL drops 12% on weak guidance",   "probs": [0.78, 0.08, 0.14]},
            {"text": "$TSLA crushes Q3, revenue +25%",     "probs": [0.06, 0.87, 0.07]},
            {"text": "Fed holds rates, awaits more data",   "probs": [0.11, 0.09, 0.80]},
        ]
        fig, axes = _dark_fig(1, 3, figsize=(13, 3))
        for ax, ep in zip(axes, examples):
            bars = ax.bar(LABEL_ORDER, ep["probs"], color=[COLORS[l] for l in LABEL_ORDER],
                          edgecolor="#152030", linewidth=0.7)
            for b, v in zip(bars, ep["probs"]):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.0%}",
                        ha="center", va="bottom", color="white", fontsize=8,
                        fontweight="bold", fontfamily="monospace")
            pred_l = LABEL_ORDER[int(np.argmax(ep["probs"]))]
            ax.set_title(f"{EMOJIS[pred_l]} {pred_l}  |  {ep['text']}", color="#88aacc",
                         fontsize=8.5, fontfamily="monospace", pad=6)
            ax.set_ylim(0, 1.1)
            _style_ax(ax)
        plt.tight_layout(pad=1.2)
        _render(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — MODEL COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    st.markdown("### Model Performance Comparison")
    st.markdown("<div class='insight-box'>Evaluated on the official validation split (2,486 tweets). <b>Macro F1</b> is the primary metric — it weights all three classes equally and prevents gaming accuracy by over-predicting the Neutral majority.</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    # Summary table
    _sumdf = pd.DataFrame([{"Model": n, "Accuracy": f"{m['Accuracy']:.4f}",
                             "Macro F1": f"{m['Macro F1']:.4f}",
                             "Bearish F1": f"{m['Bearish F1']:.4f}",
                             "Bullish F1": f"{m['Bullish F1']:.4f}",
                             "Neutral F1": f"{m['Neutral F1']:.4f}"}
                            for n, m in metrics.items()]).set_index("Model")
    st.dataframe(_sumdf.style.highlight_max(axis=0, color="#0a2810"), use_container_width=True)

    _lstm_f1 = metrics["LSTM"]["Macro F1"]
    _bert_f1 = metrics["FinBERT"]["Macro F1"]
    st.markdown(
        f"<div style='padding:10px 16px;background:#070f1a;border:1px solid #1e3050;border-radius:7px;font-size:0.82rem;color:#5a8aaa;margin:0.8rem 0;'>"
        f"FinBERT achieves <b style='color:#f0c040;'>+{_bert_f1-_lstm_f1:.4f} Macro F1</b> over the best RNN (LSTM). "
        f"Largest gains: <span style='color:#e74c3c;'>Bearish</span> (+0.14) and <span style='color:#2ecc71;'>Bullish</span> (+0.13) — "
        f"the minority classes where financial domain knowledge matters most.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Macro F1 bar + per-class heatmap
    _c1, _c2 = st.columns(2)
    _mnames  = list(metrics.keys())
    _mclrs   = ["#c0392b","#2980b9","#27ae60","#e67e22"]

    with _c1:
        st.markdown("**Macro F1 by Model**")
        fig, ax = _dark_fig(figsize=(6, 3.8))
        f1vals  = [metrics[m]["Macro F1"] for m in _mnames]
        bars    = ax.bar(_mnames, f1vals, color=_mclrs, edgecolor="#152030", linewidth=0.8)
        for b, v in zip(bars, f1vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.4f}",
                    ha="center", va="bottom", color="white", fontsize=9,
                    fontweight="bold", fontfamily="monospace")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F1", color="#4a6888", fontsize=9)
        ax.axhline(0.5, color="#1e3050", linestyle="--", linewidth=0.8)
        _style_ax(ax)
        plt.tight_layout()
        _render(fig)

    with _c2:
        st.markdown("**Per-Class F1 Heatmap**")
        hdata = np.array([[metrics[m]["Bearish F1"], metrics[m]["Bullish F1"], metrics[m]["Neutral F1"]] for m in _mnames])
        fig, ax = _dark_fig(figsize=(6, 3.8))
        sns.heatmap(hdata, annot=True, fmt=".4f", cmap="RdYlGn",
                    xticklabels=["Bearish","Bullish","Neutral"], yticklabels=_mnames,
                    vmin=0, vmax=1, linewidths=0.5, ax=ax, annot_kws={"size": 9, "family": "monospace"})
        ax.tick_params(colors="#c8d8f0", labelsize=8)
        plt.tight_layout()
        _render(fig)

    # Training curves — F1 and Loss
    for _metric, _ykey_tr, _ykey_va, _ylabel in [
        ("Macro F1 — Training Curves", "train_f1",   "valid_f1",   "Macro F1"),
        ("Loss — Training Curves",     "train_loss",  "valid_loss", "Loss"),
    ]:
        st.markdown(f"**{_metric}**")
        fig, axes = _dark_fig(1, 4, figsize=(16, 3.5))
        for ax, (mn, md), tc in zip(axes, metrics.items(), _mclrs):
            tr = md["history"][_ykey_tr]
            va = md["history"][_ykey_va]
            ax.plot(range(1, len(tr)+1), tr, color=tc, linewidth=2, label="Train")
            ax.plot(range(1, len(va)+1), va, color=tc, linewidth=2, linestyle="--", alpha=0.55, label="Val")
            if _ykey_tr == "train_f1":
                best = int(np.argmax(va)) + 1
                ax.axvline(best, color="#c8d8f0", linestyle=":", linewidth=0.9, alpha=0.5)
            ax.set_title(mn, color="#c8d8f0", fontsize=9, fontfamily="monospace", pad=5)
            ax.set_xlabel("Epoch", color="#4a6888", fontsize=7)
            ax.set_ylabel(_ylabel, color="#4a6888", fontsize=7)
            _style_ax(ax)
            ax.legend(fontsize=7, facecolor="#07101f", labelcolor="#8899bb", edgecolor="#152030",
                      loc="lower right" if _ykey_tr == "train_f1" else "upper right")
        plt.tight_layout(pad=1.2)
        _render(fig)

    st.markdown("---")

    # Detailed report + confusion matrix
    st.markdown("**Classification Report & Confusion Matrix**")
    rep_sel = st.selectbox("Select model", list(metrics.keys()), key="rep_sel")
    rep_m   = metrics[rep_sel]
    rc1, rc2 = st.columns(2)

    with rc1:
        rep_rows = [{"Class": c,
                     "Precision": f"{rep_m['report'][c]['precision']:.4f}",
                     "Recall":    f"{rep_m['report'][c]['recall']:.4f}",
                     "F1-Score":  f"{rep_m['report'][c]['f1-score']:.4f}",
                     "Support":   str(VAL_DIST[c])}
                    for c in LABEL_ORDER]
        st.dataframe(pd.DataFrame(rep_rows).set_index("Class"), use_container_width=True)
        st.markdown(
            f"<div style='margin-top:0.6rem;font-size:0.78rem;'>"
            f"<span style='color:#4a6888;'>Accuracy:</span> "
            f"<b style='color:#c8d8f0;font-family:IBM Plex Mono,monospace;'>{rep_m['Accuracy']:.4f}</b>&emsp;"
            f"<span style='color:#4a6888;'>Macro F1:</span> "
            f"<b style='color:#f0c040;font-family:IBM Plex Mono,monospace;'>{rep_m['Macro F1']:.4f}</b></div>",
            unsafe_allow_html=True,
        )

    with rc2:
        cm = np.array(rep_m["confusion"])
        fig, ax = _dark_fig(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                    linewidths=0.5, ax=ax, annot_kws={"size": 11, "family": "monospace"})
        ax.set_xlabel("Predicted", color="#4a6888", fontsize=9)
        ax.set_ylabel("True",      color="#4a6888", fontsize=9)
        ax.tick_params(colors="#c8d8f0", labelsize=8)
        plt.tight_layout()
        _render(fig)

    st.markdown("<div class='insight-box'><b>Key findings:</b><br>• <b>Simple RNN</b>: vanishing gradients — cannot learn long-range context; most tweets misclassified as Neutral.<br>• <b>LSTM / GRU</b>: gated architectures overcome this. Tied performance — tweets ≤32 tokens don't expose LSTM's extra expressiveness.<br>• <b>FinBERT</b>: bidirectional attention + financial domain pre-training gives the largest gains on Bearish and Bullish — minority classes with the subtlest language.</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — DATASET EXPLORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.markdown("### Dataset Explorer")
    st.markdown("<div class='insight-box'>Twitter Financial News Sentiment — 11,932 annotated English finance tweets across 3 classes. Official splits are used: no test set leakage.</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)

    def _dist_bar(dist: dict, title: str, ax):
        total = sum(dist.values())
        bars  = ax.bar(dist.keys(), dist.values(), color=[COLORS[l] for l in dist], edgecolor="#152030", linewidth=0.8)
        for b, (lbl, cnt) in zip(bars, dist.items()):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+total*0.012,
                    f"{cnt}\n({cnt/total*100:.1f}%)", ha="center", va="bottom",
                    color="white", fontsize=8.5, fontweight="bold", fontfamily="monospace")
        ax.set_ylim(0, max(dist.values()) * 1.3)
        ax.set_title(title, color="#c8d8f0", fontsize=10, fontfamily="monospace", pad=6)
        ax.set_ylabel("Count", color="#4a6888", fontsize=8)
        _style_ax(ax)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Class Distribution — Train (9,938)**")
        fig, ax = _dark_fig(figsize=(5.5, 3.8))
        _dist_bar(TRAIN_DIST, "Train Set", ax)
        plt.tight_layout(); _render(fig)
    with d2:
        st.markdown("**Class Distribution — Validation (2,486)**")
        fig, ax = _dark_fig(figsize=(5.5, 3.8))
        _dist_bar(VAL_DIST, "Validation Set", ax)
        plt.tight_layout(); _render(fig)

    _imb = TRAIN_DIST["Neutral"] / TRAIN_DIST["Bearish"]
    st.markdown(f"<div class='insight-box'><b>Class imbalance:</b> Neutral is <b>{_imb:.1f}×</b> more frequent than Bearish. Addressed with inverse-frequency <b>class weights</b> in CrossEntropyLoss — the minority Bearish class gets the highest training weight.</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Word / Ticker / Overlap explorer
    explore_mode = st.radio("Explore:", ["Top Words per Class","Top Tickers per Class","Shared Vocabulary (Overlap)"], horizontal=True)

    if explore_mode == "Top Words per Class":
        st.markdown("**Most Frequent Words per Sentiment Class** (stopwords removed, lemmatised)")
        fig, axes = _dark_fig(1, 3, figsize=(16, 6))
        for ax, cls in zip(axes, LABEL_ORDER):
            words = TOP_WORDS[cls][:12]
            ax.barh([w[0] for w in words][::-1], [w[1] for w in words][::-1],
                    color=COLORS[cls], edgecolor="#152030", linewidth=0.6)
            ax.set_title(f"{EMOJIS[cls]} {cls}", color="#c8d8f0", fontsize=11, fontfamily="monospace", fontweight="bold", pad=6)
            ax.set_xlabel("Frequency", color="#4a6888", fontsize=8)
            _style_ax(ax)
        plt.tight_layout(pad=2); _render(fig)
        st.markdown("<div class='insight-box'><b>Insight:</b> <i>stock</i>, <i>market</i>, <i>price</i> appear in <b>all three</b> classes — sentiment is carried by word order and context, not by individual keywords. This is why sequence models outperform bag-of-words classifiers.</div>", unsafe_allow_html=True)

    elif explore_mode == "Top Tickers per Class":
        st.markdown("**Most Mentioned Stock Tickers per Sentiment Class**")
        fig, axes = _dark_fig(1, 3, figsize=(16, 6))
        for ax, cls in zip(axes, LABEL_ORDER):
            tks = TOP_TICKERS[cls][:10]
            ax.barh([t[0] for t in tks][::-1], [t[1] for t in tks][::-1],
                    color=COLORS[cls], edgecolor="#152030", linewidth=0.6)
            ax.set_title(f"{EMOJIS[cls]} {cls}", color="#c8d8f0", fontsize=11, fontfamily="monospace", fontweight="bold", pad=6)
            ax.set_xlabel("Mentions", color="#4a6888", fontsize=8)
            _style_ax(ax)
        plt.tight_layout(pad=2); _render(fig)
        st.markdown("<div class='insight-box'>💡 <b>Insight:</b> $TSLA and $NVDA appear more in Bullish tweets; $GS and $BAC in Bearish. The same ticker (e.g. $AAPL) spans all three classes depending on the news event — ticker identity is not a reliable sentiment signal on its own.</div>", unsafe_allow_html=True)

    else:  # Overlap
        st.markdown("**Words Appearing in ALL Three Classes**")
        all_sets = [set(w[0] for w in TOP_WORDS[c]) for c in LABEL_ORDER]
        common   = all_sets[0] & all_sets[1] & all_sets[2]
        dicts    = {c: {w: cnt for w, cnt in TOP_WORDS[c]} for c in LABEL_ORDER}
        shared_df = pd.DataFrame([{"Word": w, "Bearish": dicts["Bearish"].get(w,0),
                                    "Bullish": dicts["Bullish"].get(w,0),
                                    "Neutral": dicts["Neutral"].get(w,0)}
                                   for w in common]).sort_values("Neutral", ascending=False).head(20).set_index("Word")
        st.dataframe(shared_df, use_container_width=True)
        fig, ax = _dark_fig(figsize=(12, 4))
        xs = np.arange(len(shared_df))
        for i, cls in enumerate(LABEL_ORDER):
            ax.bar(xs+i*0.28, shared_df[cls].values, 0.28, label=cls, color=COLORS[cls], edgecolor="#152030", linewidth=0.5)
        ax.set_xticks(xs+0.28)
        ax.set_xticklabels(shared_df.index, rotation=45, ha="right", fontsize=8, color="#8899bb", fontfamily="monospace")
        ax.set_ylabel("Count", color="#4a6888", fontsize=8)
        ax.set_title("Shared words across all 3 classes", color="#c8d8f0", fontsize=9, fontfamily="monospace", pad=6)
        ax.legend(fontsize=8, facecolor="#07101f", labelcolor="#8899bb", edgecolor="#152030")
        _style_ax(ax)
        plt.tight_layout(); _render(fig)
        st.markdown("<div class='insight-box'>These shared words have <b>no discriminative value</b> alone — a model relying on them will predict at chance. The model must learn sentiment from word combinations, negations, and sequence context.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Dataset at a Glance**")
    stat_cols = st.columns(4)
    for col, lbl, val in zip(stat_cols,
        ["Total Tweets","Train / Val","Imbalance Ratio","Avg Tweet Length"],
        ["11,932","9,938 / 2,486",f"{_imb:.1f}× Neutral:Bearish","~18 words"]):
        col.markdown(f"<div class='metric-tile'><div class='val' style='color:#c8dcf8;font-size:1.2rem;'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — BATCH ANALYSE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab4:
    st.markdown("### Batch Sentiment Analysis")
    st.markdown("<div class='insight-box'>Upload a CSV file with a <code>text</code> column. Every row is classified and results (label, confidence, per-class probabilities) can be downloaded as a new CSV.</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

    _demo_csv_df = pd.DataFrame({"text": [
        "$AAPL stock drops 12% after weak iPhone sales guidance",
        "$TSLA crushes Q3 earnings — revenue up 25% YoY",
        "Federal Reserve holds rates steady pending further economic data",
        "$NVDA record revenue from AI chip demand; shares surge 15% after hours",
        "$META ad revenue beats consensus; buyback programme doubled to $50B",
        "Central banks navigate unprecedented uncertainty as global growth slows",
    ]})
    st.download_button("⬇ Download sample input CSV", data=_demo_csv_df.to_csv(index=False).encode("utf-8"),
                       file_name="sample_input.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your CSV (must have a **text** column)", type=["csv"])

    if uploaded is None:
        st.markdown("---")
        st.markdown("**Expected input format:**")
        st.dataframe(_demo_csv_df, use_container_width=True)
    else:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "text" not in batch_df.columns:
            st.error("❌ CSV must contain a column named **text**.")
        else:
            st.success(f"✓ Loaded **{len(batch_df):,}** rows.")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                texts   = batch_df["text"].fillna("").astype(str).tolist()
                n       = len(texts)
                prog    = st.progress(0, text="Classifying…")
                results = []
                for i, t in enumerate(texts):
                    r = run_inference(t)
                    results.append({"text": t, "prediction": r["label"],
                                    "confidence": round(r["confidence"], 4),
                                    "prob_bearish": round(r["probs"]["Bearish"], 4),
                                    "prob_bullish": round(r["probs"]["Bullish"], 4),
                                    "prob_neutral": round(r["probs"]["Neutral"], 4)})
                    prog.progress((i+1)/n, text=f"Row {i+1}/{n}")
                prog.empty()

                if not model_ok and not _is_bert:
                    st.info("ℹ Demo mode — place `vocab.json` and `*_best.pt` files in the `models/` folder for real predictions.")

                res_df = pd.DataFrame(results)
                st.markdown("#### Results Preview")
                st.dataframe(res_df, use_container_width=True, height=280)

                # Summary metrics
                pred_counts = res_df["prediction"].value_counts().reindex(LABEL_ORDER, fill_value=0)
                avg_conf    = res_df["confidence"].mean()
                mc1,mc2,mc3,mc4 = st.columns(4)
                for mc, lbl, val, clr in zip([mc1,mc2,mc3,mc4],
                    ["Total Rows","Avg Confidence","Most Common","Least Common"],
                    [str(n), f"{avg_conf:.1%}", pred_counts.idxmax(), pred_counts.idxmin()],
                    ["#c8dcf8","#f0c040", COLORS[pred_counts.idxmax()], COLORS[pred_counts.idxmin()]]):
                    mc.markdown(f"<div class='metric-tile'><div class='val' style='color:{clr};font-size:1.3rem;'>{val}</div><div class='lbl'>{lbl}</div></div>", unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:0.8rem;'></div>", unsafe_allow_html=True)

                # Distribution charts
                vc1, vc2 = st.columns(2)
                with vc1:
                    st.markdown("**Prediction Distribution**")
                    nz = pred_counts[pred_counts > 0]
                    fig, ax = _dark_fig(figsize=(5.5, 3.5))
                    bars = ax.bar(nz.index, nz.values, color=[COLORS[l] for l in nz.index], edgecolor="#152030", linewidth=0.8)
                    for b, v in zip(bars, nz.values):
                        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, str(v),
                                ha="center", color="white", fontsize=10, fontweight="bold", fontfamily="monospace")
                    ax.set_ylabel("Count", color="#4a6888", fontsize=8)
                    _style_ax(ax); plt.tight_layout(); _render(fig)
                with vc2:
                    st.markdown("**Sentiment Mix**")
                    fig, ax = _dark_fig(figsize=(5.5, 3.5))
                    nz2 = pred_counts[pred_counts > 0]
                    ax.pie(nz2.values, labels=nz2.index, colors=[COLORS[l] for l in nz2.index],
                           autopct="%1.1f%%", startangle=90,
                           textprops={"color":"#c8d8f0","fontsize":9,"fontfamily":"monospace"},
                           wedgeprops={"edgecolor":"#07101f","linewidth":2})
                    ax.set_facecolor("#07101f"); plt.tight_layout(); _render(fig)

                # Confidence histogram
                st.markdown("**Confidence Distribution by Predicted Class**")
                fig, ax = _dark_fig(figsize=(11, 3))
                for cls in LABEL_ORDER:
                    sub = res_df[res_df["prediction"]==cls]["confidence"]
                    if len(sub):
                        ax.hist(sub, bins=20, alpha=0.7, label=cls, color=COLORS[cls], edgecolor="#152030")
                ax.set_xlabel("Confidence", color="#4a6888", fontsize=8)
                ax.set_ylabel("Count", color="#4a6888", fontsize=8)
                ax.legend(fontsize=8, facecolor="#07101f", labelcolor="#8899bb", edgecolor="#152030")
                _style_ax(ax); plt.tight_layout(); _render(fig)

                # Download button
                st.download_button("⬇ Download Predictions CSV",
                                   data=res_df.to_csv(index=False).encode("utf-8"),
                                   file_name="sentiment_predictions.csv",
                                   mime="text/csv", type="primary")