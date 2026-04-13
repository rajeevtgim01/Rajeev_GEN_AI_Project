# Sentiment/Topic Batch Classifier (Streamlit)

This project provides a Streamlit UI that:

1. Uploads a dataset (`.csv` / `.xlsx`)
2. Auto-detects a remarks/comments column
3. Processes remarks in **batches of 20**
4. For each remark:
   - Detects language and translates to English if needed
   - Summarizes
   - Maps to the best topic using TF-IDF cosine similarity
   - Predicts sentiment (Positive / Negative / Neutral)
5. Exports an `.xlsx` with:
   - `Original Remark`
   - `Summarized Remark`
   - `Topic`
   - `Sentiment`

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Notes

- First run may download the Hugging Face model weights (can take a few minutes).
- Translation uses `googletrans` (best-effort). If translation fails, the original text is used.
- Topic auto-generation uses a TF-IDF seed from the first ~200 non-empty remarks.

