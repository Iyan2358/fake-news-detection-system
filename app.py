import streamlit as st
import torch
import torch.nn as nn
import re

# ============================
# CLEAN TEXT (same as training)
# ============================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================
# LOAD VOCAB
# ============================

vocab = torch.load("vocab.pth", weights_only=False)
MAX_LEN = 400


# ============================
# MODEL DEFINITIONS
# ============================

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        return self.fc(attn_out)


# ============================
# LOAD TRAINED MODEL
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMAttentionModel(len(vocab))
model.load_state_dict(torch.load("bilstm_attention_fake_news.pth", map_location=device))
model.to(device)
model.eval()


# ============================
# ENCODE FUNCTION
# ============================

def encode(text):
    tokens = [vocab.get(w, 1) for w in text.split()]  # <UNK> = 1
    if len(tokens) > MAX_LEN:
        tokens = tokens[:MAX_LEN]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


# ============================
# PREDICT FUNCTION
# ============================

def predict(title, text):
    combined = clean_text(title) + " " + clean_text(text)
    ids = encode(combined).to(device)

    with torch.no_grad():
        output = model(ids)
        pred_idx = torch.argmax(output, dim=1).item()

    # Manual mapping: 0 = REAL, 1 = FAKE
    return "FAKE" if pred_idx == 1 else "REAL"


# ============================
# STREAMLIT UI
# ============================

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System (Bi-LSTM + Attention)")
st.write("Enter a news **title** and **content** to check if it's REAL or FAKE.")

title_input = st.text_input("News Title")
text_input = st.text_area("News Content")

if st.button("Analyze"):
    if title_input.strip() == "" or text_input.strip() == "":
        st.error("Please enter both title and text.")
    else:
        prediction = predict(title_input, text_input)

        if prediction == "FAKE":
            st.error("🚨 The system predicts: **FAKE NEWS**")
        else:
            st.success("✅ The system predicts: **REAL NEWS**")
