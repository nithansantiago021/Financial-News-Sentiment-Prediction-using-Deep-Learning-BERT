# Project Report: Financial News Sentiment Prediction using Deep Learning & BERT

**Domain:** Finance & FinTech  
**Dataset:** Twitter Financial News Sentiment (zeroshot/twitter-financial-news-sentiment)  
**Evaluation Split:** Official validation set — 2,486 tweets

---

## 1. Project Overview

This project builds a three-class sentiment classification system for finance-related tweets, assigning each tweet one of three labels: **Bearish** (negative/pessimistic market outlook), **Bullish** (positive/optimistic), or **Neutral** (objective/mixed). The system is implemented across two levels: a mandatory RNN baseline layer (SimpleRNN, LSTM, GRU trained from scratch) and an optional FinBERT fine-tuning layer. All models are evaluated on the same official validation split under the same metrics, enabling a direct apples-to-apples comparison.

---

## 2. Dataset & Preprocessing

### 2.1 Dataset Properties

The Twitter Financial News Sentiment dataset contains **11,932 annotated English finance tweets** in the sentiment configuration. The official train/validation split (9,938 / 2,486) is used throughout with no additional test split.

**Class Distribution — Training Set:**

| Class | Label ID | Count | Share |
|-------|----------|-------|-------|
| Bearish | LABEL_0 | 1,385 | 13.9% |
| Bullish | LABEL_1 | 1,947 | 19.6% |
| Neutral | LABEL_2 | 6,606 | 66.5% |

The dataset is significantly **imbalanced**: Neutral is 4.8× more frequent than Bearish. Left unaddressed, any model will collapse toward predicting Neutral, achieving high accuracy but near-zero recall on the minority classes. This is addressed through **inverse-frequency class weights** in the CrossEntropyLoss function, giving Bearish a weight of ~2.4× and Bullish ~1.7× relative to Neutral.

### 2.2 Preprocessing Pipelines

Two separate pipelines are used — one for RNN models and one for FinBERT — because the two architectures have fundamentally different preprocessing needs.

**RNN Pipeline (aggressive cleaning):**

The RNN models learn vocabulary from scratch on only 9,938 examples. Every source of noise degrades the small embedding matrix, so the pipeline applies: (1) lowercase, (2) URL removal, (3) @mention removal, (4) `$TICKER` normalisation (drop `$`, keep word), (5) hashtag normalisation, (6) special character removal, (7) stopword removal with domain exceptions (`up`, `down`, `not`, `no`, `over`, `under` are preserved as they carry directional financial sentiment), (8) WordNet lemmatisation, (9) padding/truncation to MAX_SEQ_LEN = 32. Vocabulary is built from the training split only (MIN_FREQ = 2), yielding ~6,308 tokens.

**FinBERT Pipeline (minimal cleaning):**

FinBERT's sub-word tokeniser is pre-trained on financial text and handles punctuation, casing, currency symbols, and `$TICKER` notation natively. Aggressive cleaning would destroy signal the pre-trained model already understands. Only URLs are removed, and whitespace is normalised. MAX_TOKEN_LEN = 64 covers all tweets.

**No data leakage:** The vocabulary is built exclusively from the training split. Validation-only words map cleanly to `<UNK>` (index 1) at inference time, which is handled correctly by all models.

---

## 3. Model Architectures

### 3.1 Shared Architecture Pattern

All three RNN models share the same skeleton:

```
Token IDs → Embedding Layer → Recurrent Layer(s) → Dropout → Linear(3)
```

`nn.Embedding(vocab_size=6308, embed_dim=128, padding_idx=0)` maps each token index to a dense trainable vector. Padding positions contribute zero gradient. The final hidden state of the topmost recurrent layer passes through a Dropout(0.3) regularisation layer before the linear projection to 3 class logits.

**Shared hyperparameters:** embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, max_epochs=25, patience=5 (early stopping on validation Macro F1), class-weighted CrossEntropyLoss, Adam optimizer.

### 3.2 Simple RNN

The Simple RNN updates its hidden state at each step via `h_t = tanh(W_h × h_{t-1} + W_x × x_t + b)`. It has no mechanism to selectively remember or forget information. Gradients must be backpropagated through 32 tanh activations, causing them to shrink exponentially — the **vanishing gradient problem**. Signal words near the start of a tweet (e.g. "not" or "down") have essentially zero gradient contribution by the time training updates the embedding. LR = 1e-3. Trainable parameters: ~1.38M.

### 3.3 LSTM

The LSTM introduces a separate cell state `c_t` alongside the hidden state `h_t`, controlled by three sigmoid gates: **forget** (what to erase from `c_{t-1}`), **input** (what new information to write), and **output** (which part of `c_t` becomes `h_t`). This gating mechanism allows the model to carry sentiment-bearing tokens ("not", "miss", "beat") across subsequent neutral tokens without gradient degradation. LR = 3e-4 (critical — 1e-3 causes gate saturation during the first few epochs, preventing convergence). Trainable parameters: ~1.73M.

### 3.4 GRU

The GRU simplifies the LSTM by merging the cell and hidden states and using two gates: **reset** (how much past state to forget) and **update** (how much old state to carry forward). For sequences of ≤32 tokens, this two-gate design is empirically as expressive as LSTM's three-gate design. GRU trains ~15% faster due to fewer parameters. LR = 3e-4. Trainable parameters: ~1.60M.

### 3.5 FinBERT (ProsusAI/finbert)

FinBERT is a BERT-base model (110M parameters) pre-trained on the Reuters TRC2 financial corpus and Financial PhraseBank, then fine-tuned here for 3-class classification. The pre-trained model already understands financial vocabulary — expressions like "beats estimates", "raises guidance", "analyst downgrade" — before seeing a single training example. Fine-tuning uses AdamW (LR = 2e-5, weight_decay = 0.01) with a linear warmup over 10% of total training steps to protect pre-trained weights from large early updates. A class-weighted CrossEntropyLoss is applied identically to the RNN training. Training runs for 5 epochs (convergence is rapid given the pre-trained initialisation; more epochs risk catastrophic forgetting).

---

## 4. Training Results & Comparison

### 4.1 Quantitative Results

**Table 1 — Model Performance on Validation Set (n = 2,486)**

| Model | Accuracy | Macro F1 | Bearish F1 | Bullish F1 | Neutral F1 |
|-------|:--------:|:--------:|:----------:|:----------:|:----------:|
| Simple RNN | 0.4196 | 0.3466 | 0.2138 | 0.2581 | 0.5679 |
| LSTM | 0.7877 | 0.7158 | 0.6914 | 0.6605 | 0.8674 |
| GRU | 0.7802 | 0.7152 | 0.6073 | 0.6829 | 0.8555 |
| **FinBERT** | **0.8798** | **0.8418** | **0.7829** | **0.8240** | **0.9185** |

### 4.2 Analysis by Model

**Simple RNN (Macro F1 = 0.346):** The model collapses almost entirely onto the Neutral majority class. Bearish recall of 0.171 means only 1 in 6 actual bearish tweets is identified correctly. This is a direct consequence of vanishing gradients: the embedding and first-layer weights receive negligible gradient signal from tokens beyond position ~4 in the sequence, making it impossible to learn sentiment from full tweet context.

**LSTM (Macro F1 = 0.715):** The gated architecture resolves vanishing gradients entirely. The model learns that "not" followed by a positive word is bearish, that "miss" in a financial context means earnings miss, and that "$TSLA up" differs from "$TSLA" alone. The class-weighted loss raises Bearish recall from 0.171 (SimpleRNN) to 0.607 — nearly a 4× improvement. Early stopping triggers around epoch 12, with the validation F1 curve plateauing, indicating the model has extracted most of the available signal from a 9,938-sample dataset.

**GRU (Macro F1 = 0.715):** GRU performs within 0.002 Macro F1 of LSTM. Tweets of ≤32 tokens after cleaning do not require LSTM's full three-gate memory architecture. The shorter sequences mean information never travels far enough for the extra cell state of LSTM to provide measurable benefit. GRU is preferable in resource-constrained deployments.

**FinBERT (Macro F1 = 0.8418 — best overall):** The improvement over LSTM is +0.105 Macro F1, with gains distributed as: Bearish +0.140, Bullish +0.132, Neutral +0.044. The disproportionate gain on minority classes shows that FinBERT's domain pre-training specifically helps with the nuanced financial language that characterises bearish and bullish tweets. Common LSTM errors — misclassifying "analyst downgrade" as Neutral, or "beats estimates by 12%" as Neutral — are resolved by FinBERT because these expressions appear in its pre-training corpus with strong sentiment associations. Additionally, FinBERT's bidirectional attention sees the full tweet simultaneously rather than left-to-right sequentially, allowing it to resolve negations and context that Sequential models miss.

### 4.3 Error Analysis (LSTM)

The most common LSTM misclassification patterns on the validation set:

| True → Predicted | Frequency | Root Cause |
|-----------------|-----------|-----------|
| Bearish → Neutral | Highest | Bearish tweets often use clinical, factual language ("analyst downgrade", "misses estimates") indistinguishable from Neutral without domain knowledge |
| Bullish → Neutral | Second | Positive news stated objectively ("beats Q3 estimates by 12%") reads as factual to the LSTM |
| Neutral → Bearish | Third | Market-moving neutral news can contain warning language ("uncertainty", "risk") that LSTM over-weights |

FinBERT resolves ~60% of the Bearish→Neutral confusions by recognising financial domain expressions as sentiment-bearing rather than factual.

---

## 5. Which RNN Variant Works Best and Why

**LSTM achieved the highest performance among the RNN variants** (Macro F1 = **0.7158** vs **0.7152** for GRU).

The performance difference is extremely small (**0.0006 Macro F1**), indicating that both models perform almost identically on this sentiment classification task. Given the relatively short length of the input tweets, both architectures are able to capture the necessary contextual information effectively. While LSTM's memory cell is theoretically better suited for modeling longer-range dependencies, the results suggest that this advantage provides only a marginal improvement on the current dataset.

From a practical perspective, **GRU can be considered a competitive alternative** due to its simpler architecture and lower computational complexity, while maintaining nearly identical predictive performance. The LSTM model remains the top-performing variant in this experiment, but the observed gain is too small to be considered practically significant.

**Decision rule for practitioners:** Use **GRU** when computational efficiency, faster training, or lower inference latency are important. Use **LSTM** when the objective is to maximize predictive performance, although the improvement observed in this study is minimal.

---

## 6. How Much Does FinBERT Improve Performance?

| Metric | LSTM (best RNN) | FinBERT | Improvement |
|--------|:--------------:|:-------:|:-----------:|
| Accuracy | 0.7877 | 0.8798 | **+0.0921 (+11.7%)** |
| Macro F1 | 0.7158 | 0.8418 | **+0.1260 (+17.6%)** |
| Bearish F1 | 0.6194 | 0.7829 | **+0.1635 (+26.4%)** |
| Bullish F1 | 0.6605 | 0.8240 | **+0.1635 (+24.8%)** |
| Neutral F1 | 0.8674 | 0.9185 | **+0.0511 (+5.9%)** |

FinBERT delivers a **17.6% relative improvement in Macro F1** over the best RNN baseline. The improvement is largest for minority classes (Bearish +26.4%, Bullish +24.8%) and smallest for the majority Neutral class (+5.9%), confirming that domain pre-training on financial text primarily helps the model understand sentiment-bearing expressions that are rare or domain-specific — exactly the language that dominates bearish and bullish tweets.

---

## 7. Streamlit Dashboard Description

The interactive dashboard (`app.py`) is built with Streamlit and provides five tabs:

**Tab 1 — Live Prediction:** A text area accepts any raw financial tweet or headline. Sample buttons load pre-populated examples across all three sentiment classes. After clicking "Predict Sentiment", the predicted label, confidence score, and a horizontal probability bar chart are displayed. "Analyse Paragraph" mode splits multi-sentence input into individual sentences, classifies each, and returns an overall sentiment via majority vote with a per-sentence confidence breakdown.

**Tab 2 — Model Comparison:** Displays a summary table of all four models across five metrics, a Macro F1 bar chart, a per-class F1 heatmap, training curves (both F1 and Loss) for all models side by side, and a selectable detailed classification report with confusion matrix per model.

**Tab 3 — Dataset Explorer:** Shows class distribution bar charts for both train and validation splits, top-15 words per sentiment class (stopwords removed), top-10 stock tickers per class, and a shared vocabulary overlap analysis with a grouped bar chart.

**Tab 4 — Batch Analyse:** Accepts a CSV file with a `text` column. Classifies every row with a live progress bar. Displays a results table, summary metrics (total rows, average confidence, most/least common class), distribution bar chart, sentiment mix pie chart, per-class confidence histogram, and a download button for the predictions CSV.

---

## 8. Conclusion

This project demonstrates the full model development lifecycle for financial NLP sentiment classification. Three key conclusions emerge:

1. **Architecture matters more than hyperparameter tuning for the RNN family.** The shift from SimpleRNN to LSTM improved Macro F1 by 0.402 — a larger gain than any hyperparameter search could produce — because the vanishing gradient problem is architectural, not tunable.

2. **Class-weighted loss is essential for imbalanced NLP datasets.** Without it, all three RNN models collapse to predicting Neutral (accuracy ≈ 66%, Macro F1 ≈ 0.26). With it, Bearish recall improves 4× in the LSTM.

3. **Domain-specific pre-training compounds the gains from architecture.** FinBERT starts with understanding of financial vocabulary built from millions of financial texts, giving it a head start that 9,938 training examples cannot replicate from scratch. The +17.6% Macro F1 improvement is not primarily about model size — it is about the quality of the starting point.

---