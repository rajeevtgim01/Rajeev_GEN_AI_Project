import io
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Batch Text Classifier Bot", layout="wide")

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def resolve_gemini_api_key(sidebar_key: str = "") -> Optional[str]:
    """Prefer sidebar input, then env GEMINI_API_KEY, then Streamlit secrets. Never hardcode keys in source."""
    s = (sidebar_key or "").strip()
    if s:
        return s
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return None


def configure_gemini(api_key: str) -> None:
    import google.generativeai as genai

    genai.configure(api_key=api_key)


def make_gemini_model(model_name: str):
    import google.generativeai as genai

    return genai.GenerativeModel(
        model_name=model_name.strip() or DEFAULT_GEMINI_MODEL,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )


def _gemini_response_text(response: Any) -> str:
    try:
        t = getattr(response, "text", None)
        return (t or "").strip()
    except Exception:
        return ""


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    return raw.strip()


def _snap_sentiment(label: str) -> str:
    x = (label or "").strip().lower()
    if x == "positive":
        return "Positive"
    if x == "negative":
        return "Negative"
    return "Neutral"


def _snap_topic(label: str, topics: List[str]) -> str:
    lab = (label or "").strip()
    if not topics:
        return lab or "Other"
    low = lab.lower()
    for t in topics:
        if t.lower() == low:
            return t
    for t in topics:
        tl = t.lower()
        if low in tl or tl in low:
            return t
    for t in topics:
        if t.lower() == "other":
            return t
    return topics[0]


def _gemini_generate_content(model, prompt: str) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            text = _gemini_response_text(response)
            if text:
                return text
            last_err = RuntimeError("Empty response from Gemini (blocked or no text).")
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)
    raise last_err or RuntimeError("Gemini request failed.")


def gemini_suggest_topics(model, remarks_sample: List[str], count: int) -> List[str]:
    """One API call to propose topic labels when the user did not supply any."""
    lines = [normalize_text(r) for r in remarks_sample if normalize_text(r)][:45]
    if not lines:
        return []
    prompt = (
        f"You design topic labels for classifying short customer feedback.\n"
        f"Return JSON only with this shape: {{\"topics\": [string, ...]}}\n"
        f"The array must have exactly {count} distinct, concise topic names (2-5 words each), no numbering.\n\n"
        f"Sample feedback lines:\n"
        + "\n".join(f"- {x[:500]}" for x in lines)
    )
    try:
        raw = _strip_json_fences(_gemini_generate_content(model, prompt))
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError, RuntimeError):
        return []
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        return []
    out = []
    seen = set()
    for t in topics:
        t = normalize_text(str(t))
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[: max(3, count)]


def gemini_process_batch(
    model,
    batch: List[str],
    topics: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    One Gemini call per batch: translate-if-needed + summarize (English), topic (from list), sentiment.
    """
    n = len(batch)
    topic_lines = "\n".join(f"  - {t}" for t in topics)
    payload = [{"index": i, "text": (batch[i] or "")[:8000]} for i in range(n)]

    prompt = f"""You process feedback rows in one batch.

For EACH row:
1) If the text is not in English, mentally translate to English first.
2) summarized_remark: short meaningful English summary (1-2 sentences). If the original is already short English, you may lightly shorten it.
3) topic: MUST be exactly one string copied verbatim from ALLOWED TOPICS below (same spelling and casing as listed).
4) sentiment: exactly one of: Positive, Negative, Neutral

ALLOWED TOPICS (choose only from this list):
{topic_lines}

INPUT_ROWS_JSON:
{json.dumps(payload, ensure_ascii=False)}

OUTPUT: JSON only — a JSON array of {n} objects, one per input index 0..{n - 1}, sorted by "index" ascending.
Each object: {{"index": int, "summarized_remark": str, "topic": str, "sentiment": str}}
Do not include markdown fences or commentary.
"""

    raw = _strip_json_fences(_gemini_generate_content(model, prompt))
    arr = json.loads(raw)
    if not isinstance(arr, list):
        raise ValueError("Gemini returned non-array JSON.")

    by_idx: Dict[int, Dict[str, Any]] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index", -1))
        except (TypeError, ValueError):
            continue
        by_idx[idx] = item

    summaries: List[str] = []
    tops: List[str] = []
    sents: List[str] = []
    for i in range(n):
        row = by_idx.get(i, {})
        orig = (batch[i] or "").strip()
        summ = normalize_text(str(row.get("summarized_remark", "")))
        if not summ and orig:
            summ = orig[:200]
        summaries.append(summ)
        tops.append(_snap_topic(str(row.get("topic", "")), topics))
        sents.append(_snap_sentiment(str(row.get("sentiment", ""))))

    return summaries, tops, sents


# --- Local processing (no API key): extractive summary + TF-IDF topics + VADER sentiment ---


def extractive_summarize(text: str, max_chars: int = 220) -> str:
    """Lightweight summary: first sentence(s), trimmed — no LLM."""
    t = normalize_text(text)
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = parts[0] if parts else t
    if len(parts) > 1 and len(out) + 1 + len(parts[1]) <= max_chars:
        out = f"{out} {parts[1]}"
    if len(out) > max_chars:
        out = out[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return out


def fit_topic_vectorizer(topics: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    topic_vecs = vec.fit_transform([normalize_text(t) for t in topics])
    return vec, topic_vecs


def map_topics_tfidf(vec, topic_vecs, summaries: List[str]) -> List[int]:
    from sklearn.metrics.pairwise import cosine_similarity

    summary_vecs = vec.transform([normalize_text(s) for s in summaries])
    sims = cosine_similarity(summary_vecs, topic_vecs)
    sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
    return sims.argmax(axis=1).tolist()


@st.cache_resource
def get_vader_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def vader_sentiment_labels(texts: List[str]) -> List[str]:
    """Positive / Negative / Neutral from VADER compound score."""
    analyzer = get_vader_analyzer()
    out: List[str] = []
    for t in texts:
        if not normalize_text(t or ""):
            out.append("Neutral")
            continue
        compound = analyzer.polarity_scores(t)["compound"]
        if compound >= 0.05:
            out.append("Positive")
        elif compound <= -0.05:
            out.append("Negative")
        else:
            out.append("Neutral")
    return out


def local_process_batch(
    batch: List[str],
    topics: List[str],
    vec,
    topic_vecs,
) -> Tuple[List[str], List[str], List[str]]:
    summaries = [extractive_summarize(t) for t in batch]
    idxs = map_topics_tfidf(vec, topic_vecs, summaries)
    tops = [topics[i] if topics and 0 <= i < len(topics) else (topics[0] if topics else "Other") for i in idxs]
    sents = vader_sentiment_labels(summaries)
    return summaries, tops, sents


# Column headers we never use for automatic remarks detection (topic templates, output fields, blank Excel cols).
_REMARKS_AUTO_SKIP_EXACT = frozenset(
    {
        "topics",
        "topic",
        "summarization",
        "summary",
        "sentiment",
    }
)
_UNNAMED_COL_RE = re.compile(r"^unnamed:\s*\d+\s*$", re.IGNORECASE)


def is_skipped_remarks_header_for_auto_detect(col_name: str) -> bool:
    """True if this column name should be ignored when auto-detecting remarks."""
    raw = str(col_name).strip()
    low = raw.lower()
    if low in _REMARKS_AUTO_SKIP_EXACT:
        return True
    if _UNNAMED_COL_RE.match(low):
        return True
    return False


def is_skipped_remarks_header_strict(col_name: str) -> bool:
    """Skip topic/summary/sentiment labels only (still allow Unnamed:* for default pick)."""
    low = str(col_name).strip().lower()
    return low in _REMARKS_AUTO_SKIP_EXACT


def _remarks_keyword_score(low: str) -> int:
    """Higher = better match for a free-text / feedback column."""
    if is_skipped_remarks_header_for_auto_detect(low):
        return -1
    # Prefer specific phrases, then substrings.
    phrases = (
        "free text",
        "open text",
        "verbatim",
        "patient feedback",
        "customer feedback",
    )
    for p in phrases:
        if p in low:
            return 100 + len(p)
    keys = (
        "remark",
        "comment",
        "feedback",
        "review",
        "description",
        "narrative",
        "concern",
        "suggestion",
        "complaint",
        "experience",
    )
    best = 0
    for k in keys:
        if k in low:
            best = max(best, 50 + len(k))
    return best


def _column_long_text_score(series: pd.Series) -> float:
    """Prefer columns that look like long free-text, not short labels."""
    s = series.dropna().astype(str).map(normalize_text)
    s = s[s.astype(bool)]
    if s.empty:
        return 0.0
    lengths = s.str.len()
    longish = (lengths > 40).sum()
    return float(longish * 10_000 + lengths.sum())


def detect_remarks_column(df: pd.DataFrame) -> Tuple[str, List[str]]:
    """
    Pick a remarks column: keyword match (skipping topic/summary/sentiment/Unnamed:*), else heuristic on allowed columns.
    """
    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    eligible = [c for c in cols if not is_skipped_remarks_header_for_auto_detect(c)]
    if not eligible:
        return "", cols

    # 1) Keyword / phrase match among eligible columns
    best_c = None
    best_score = 0
    for c in eligible:
        sc = _remarks_keyword_score(lowered[c])
        if sc > best_score:
            best_score = sc
            best_c = c
    if best_c is not None and best_score > 0:
        return best_c, cols

    # 2) Heuristic: most "long text" among eligible
    best_c2 = None
    best_h = -1.0
    for c in eligible:
        h = _column_long_text_score(df[c])
        if h > best_h:
            best_h = h
            best_c2 = c
    if best_c2 is not None and best_h > 0:
        return best_c2, cols

    return "", cols


def default_remarks_column_for_selectbox(df: pd.DataFrame) -> str:
    """
    If strict auto-detect fails, suggest a column for the dropdown: best long-text column
    among names that are not Topics/Summarization/Sentiment (Unnamed:* allowed).
    """
    cols = list(df.columns)
    detected, _ = detect_remarks_column(df)
    if detected:
        return detected

    candidates = [c for c in cols if not is_skipped_remarks_header_strict(c)]
    if not candidates:
        return cols[0] if cols else ""

    best_c = None
    best_h = -1.0
    for c in candidates:
        h = _column_long_text_score(df[c])
        if h > best_h:
            best_h = h
            best_c = c
    return best_c if best_c is not None else cols[0]


def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Try common encodings; Windows Excel exports often use cp1252 (e.g. byte 0x92)."""
    encodings = ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1")
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err or UnicodeDecodeError("unknown", b"", 0, 1, "Could not decode CSV")


def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    bio = io.BytesIO(file_bytes)

    if name.endswith(".csv"):
        return read_csv_bytes(file_bytes)
    if name.endswith(".xlsx"):
        try:
            return pd.read_excel(bio, engine="openpyxl")
        except ImportError as e:
            raise ValueError("Reading .xlsx requires openpyxl. Run: pip install openpyxl") from e
    if name.endswith(".xls"):
        try:
            return pd.read_excel(bio)
        except ImportError as e:
            raise ValueError("Reading .xls may require xlrd. Try saving as .xlsx or: pip install xlrd") from e

    raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_topics_from_text(topics_text: str) -> List[str]:
    raw = topics_text.strip()
    if not raw:
        return []

    # Allow comma or newlines as separators.
    parts = re.split(r",|\n", raw)
    topics = [normalize_text(p) for p in parts]
    topics = [t for t in topics if t]

    # Deduplicate while preserving order
    seen = set()
    out = []
    for t in topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def load_topics_from_upload(uploaded_file) -> List[str]:
    df = read_uploaded_dataset(uploaded_file)
    cols = list(df.columns)
    lowered = {c: str(c).strip().lower() for c in cols}

    topic_col = None
    for c in cols:
        if lowered[c] == "topic" or lowered[c] == "topics":
            topic_col = c
            break
    if topic_col is None:
        topic_col = cols[0] if cols else None

    if topic_col is None:
        return []

    topics = df[topic_col].dropna().astype(str).map(normalize_text).tolist()
    return [t for t in topics if t]


def generate_topics_from_remarks(remarks: List[str], topic_count: int = 10, seed_max_rows: int = 200) -> List[str]:
    """
    Topic generation via TF-IDF top terms.
    Memory-friendly and avoids expensive embedding models.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    seed = [normalize_text(r) for r in remarks if normalize_text(r)]
    if not seed:
        return ["Other"]

    seed = seed[:seed_max_rows]

    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
    )
    X = vec.fit_transform(seed)
    # Mean relevance across docs
    scores = X.mean(axis=0).A1
    feature_names = vec.get_feature_names_out()

    top_idx = scores.argsort()[::-1][: max(1, topic_count)]
    topics = [normalize_text(feature_names[i]) for i in top_idx if feature_names[i]]

    # Keep topics unique and cap length
    out = []
    seen = set()
    for t in topics:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out[: max(1, topic_count)]


SENTIMENT_DONUT_COLORS = {
    "Positive": "#1fa37a",
    "Neutral": "#8b9aab",
    "Negative": "#d94c4c",
}


def build_download_xlsx(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    bio.seek(0)
    return bio.read()


def render_output_analysis(df: pd.DataFrame) -> None:
    """
    In-app charts from processed data only. Does not modify df or the Excel export.
    """
    need = {"Sentiment", "Topic", "Original Remark", "Summarized Remark"}
    if not need.issubset(set(df.columns)):
        return

    st.subheader("Analysis")
    st.caption("Charts are computed from the preview dataset above. The downloaded file is unchanged.")

    n = len(df)
    sent = df["Sentiment"].astype(str)
    pos_n = (sent == "Positive").sum()
    neg_n = (sent == "Negative").sum()
    neu_n = (sent == "Neutral").sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows analyzed", f"{n:,}")
    m2.metric("Positive", f"{pos_n:,}", f"{100 * pos_n / n:.1f}%" if n else "—")
    m3.metric("Negative", f"{neg_n:,}", f"{100 * neg_n / n:.1f}%" if n else "—")
    m4.metric("Neutral", f"{neu_n:,}", f"{100 * neu_n / n:.1f}%" if n else "—")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Sentiment distribution**")
        s_counts = df["Sentiment"].astype(str).value_counts()
        pref_order = ["Positive", "Neutral", "Negative"]
        donut_labels: List[str] = []
        donut_values: List[int] = []
        for lab in pref_order:
            if lab in s_counts.index:
                donut_labels.append(lab)
                donut_values.append(int(s_counts.loc[lab]))
        for lab in s_counts.index:
            ls = str(lab)
            if ls not in donut_labels:
                donut_labels.append(ls)
                donut_values.append(int(s_counts.loc[lab]))
        donut_colors = [SENTIMENT_DONUT_COLORS.get(l, "#9b59b6") for l in donut_labels]
        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=donut_labels,
                    values=donut_values,
                    hole=0.56,
                    sort=False,
                    marker=dict(colors=donut_colors, line=dict(color="rgba(255,255,255,0.9)", width=2)),
                    texttemplate="<b>%{label}</b><br>%{percent:.1%}",
                    textposition="outside",
                    hovertemplate="<b>%{label}</b><br>Rows: %{value}<br>%{percent:.1%}<extra></extra>",
                )
            ]
        )
        fig_donut.update_layout(
            height=420,
            margin=dict(t=28, b=28, l=28, r=28),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        st.markdown("**Top topics** (up to 15)")
        t_counts = df["Topic"].astype(str).value_counts().head(15)
        top_df = t_counts.reset_index()
        top_df.columns = ["Topic", "count"]
        fig_topics = go.Figure(
            data=[
                go.Bar(
                    x=top_df["Topic"],
                    y=top_df["count"],
                    marker=dict(
                        color=top_df["count"],
                        colorscale=[[0, "#6eb5ff"], [1, "#1e5f99"]],
                        showscale=False,
                    ),
                    hovertemplate="<b>%{x}</b><br>Rows: %{y}<extra></extra>",
                )
            ]
        )
        fig_topics.update_layout(
            height=420,
            margin=dict(t=28, b=120, l=56, r=24),
            xaxis_title="",
            yaxis_title="Row count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False),
            xaxis=dict(tickangle=-32),
        )
        st.plotly_chart(fig_topics, use_container_width=True)

    st.markdown("**Original remark length (characters)**")
    orig_lens = df["Original Remark"].astype(str).str.len()
    sum_lens = df["Summarized Remark"].astype(str).str.len()
    bin_edges = [0, 50, 100, 200, 350, 500, 1000, 30000]
    hist_o, edges = np.histogram(orig_lens, bins=bin_edges)
    len_labels = [f"{int(edges[i])}–{int(edges[i + 1])}" for i in range(len(hist_o))]
    fig_len = go.Figure(
        data=[
            go.Bar(
                x=len_labels,
                y=hist_o,
                marker=dict(color="#7d5ba6", line=dict(width=0)),
                hovertemplate="Range: %{x}<br>Rows: %{y}<extra></extra>",
            )
        ]
    )
    fig_len.update_layout(
        height=380,
        margin=dict(t=24, b=80, l=56, r=24),
        xaxis_title="Length range (chars)",
        yaxis_title="Rows",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)", zeroline=False),
        xaxis=dict(tickangle=-25),
    )
    st.plotly_chart(fig_len, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"Average original length: **{orig_lens.mean():.0f}** chars")
    with col_b:
        st.caption(f"Average summary length: **{sum_lens.mean():.0f}** chars")

    st.markdown("**Sentiment × topic (counts)**")
    ct = pd.crosstab(df["Topic"].astype(str), df["Sentiment"].astype(str))
    st.dataframe(ct, use_container_width=True)


def main():
    st.title("Text Classifier Chat Bot")

    st.sidebar.header("Processing")
    engine = st.sidebar.radio(
        "Engine",
        options=["local", "gemini"],
        format_func=lambda x: "Local — no API key (basic)" if x == "local" else "Google Gemini (best quality)",
        help="Local uses extractive summaries + keyword topic matching + VADER sentiment. Gemini needs a free API key.",
    )

    gemini_model = DEFAULT_GEMINI_MODEL
    gemini_key_input = ""
    if engine == "gemini":
        st.sidebar.subheader("Gemini settings")
        gemini_model = st.sidebar.text_input(
            "Model name",
            value=DEFAULT_GEMINI_MODEL,
            help="e.g. gemini-2.0-flash or gemini-1.5-flash",
        )
        gemini_key_input = st.sidebar.text_input(
            "API key (optional if GEMINI_API_KEY is set)",
            type="password",
            help="Environment variable or .streamlit/secrets.toml — do not commit keys to git.",
        )

    with st.sidebar.expander("Get a free Gemini API key (optional)"):
        st.markdown(
            "1. Open **[Google AI Studio](https://aistudio.google.com/apikey)** and sign in with Google.\n"
            "2. Click **Create API key** and copy it.\n"
            "3. Paste in the sidebar above, set env `GEMINI_API_KEY`, or add to `.streamlit/secrets.toml`.\n\n"
            "Free tier is enough for learning."
        )

    st.markdown(
        "Upload a dataset (`.csv` / `.xlsx`). Remarks are processed in **batches of 20**: "
        "**summarized** text, **topic** (your list or auto-generated labels), and **sentiment**. "
        "Choose **Local** if you have no API key, or **Gemini** for smarter summaries and multilingual handling. "
        "Download **.xlsx** with Original Remark, Summarized Remark, Topic, Sentiment."
    )

    uploaded_dataset = st.file_uploader("Upload File", type=["csv", "xlsx"], key="dataset_uploader")

    st.subheader("Topics (Input #2)")
    topics_text = st.text_area(
        "Enter topics (comma-separated). Leave blank to auto-generate.",
        height=90,
        placeholder="Topic A, Topic B, Topic C",
        key="topics_text",
    )
    uploaded_topic_file = st.file_uploader("Or upload a topic file (must contain a `Topic` column).", type=["csv", "xlsx"], key="topic_uploader")

    topic_count = st.number_input("Auto-generate topic count (when topics are empty)", min_value=3, max_value=30, value=10, step=1)

    batch_size = 20  # Strict requirement

    start = st.button("Generate Output", type="primary", disabled=uploaded_dataset is None)

    remark_col_selected: Optional[str] = None
    if uploaded_dataset is not None:
        try:
            raw_df_preview = read_uploaded_dataset(uploaded_dataset)
            cols = list(raw_df_preview.columns)
            if not cols:
                st.error("The file has no columns (empty header row?).")
            else:
                auto_col, _ = detect_remarks_column(raw_df_preview)
                default_col = default_remarks_column_for_selectbox(raw_df_preview)
                default_idx = cols.index(default_col) if default_col in cols else 0
                default_idx = min(max(default_idx, 0), len(cols) - 1)

                if auto_col:
                    st.success(
                        f"Auto-detected remarks column: `{auto_col}` "
                        "(headers like Topics, Summarization, Sentiment, and Unnamed:* are skipped for auto-detect)."
                    )
                else:
                    st.info(
                        "No remarks column matched by name. Pick the column that contains feedback text below. "
                        "Auto-detect skips: Topics, Topic, Summarization, Summary, Sentiment, and Unnamed:*."
                    )

                remark_col_selected = st.selectbox(
                    "Remarks / feedback column",
                    options=cols,
                    index=default_idx,
                    key="remarks_column_pick",
                    help="Use this if your file uses a non-standard column name or feedback is in an Unnamed column.",
                )
        except Exception as e:
            st.error(f"Failed to read uploaded dataset: {e}")

    if start:
        # Clear previous run outputs
        for k in list(st.session_state.keys()):
            if k.startswith("output_"):
                del st.session_state[k]

        # Read dataset
        try:
            df_raw = read_uploaded_dataset(uploaded_dataset)
        except Exception as e:
            st.error(f"Failed to read uploaded dataset: {e}")
            return

        remark_col = remark_col_selected or st.session_state.get("remarks_column_pick")
        if remark_col is None:
            st.error("Could not determine remarks column. Re-upload the file or refresh the page.")
            return

        if remark_col not in df_raw.columns:
            st.error(f"Selected column `{remark_col}` is not in this file.")
            return

        # Keep only non-empty/null remarks
        remarks_series = df_raw[remark_col]
        remarks_series = remarks_series.dropna()
        remarks_series = remarks_series[remarks_series.astype(str).map(normalize_text).astype(bool)]

        if remarks_series.empty:
            st.error("No non-empty remarks found after filtering.")
            return

        # Copy to a clean frame so we don't lose alignment with processed outputs
        df_clean = df_raw.loc[remarks_series.index].copy().reset_index(drop=True)
        df_clean["__original_remark"] = df_clean[remark_col].astype(str).map(normalize_text)

        originals = df_clean["__original_remark"].tolist()
        total = len(originals)
        st.write(f"Total non-empty remarks to process: `{total}`")

        # Topics: user-provided > uploaded file > auto-generated
        topics: List[str] = []
        if uploaded_topic_file is not None:
            try:
                topics = load_topics_from_upload(uploaded_topic_file)
            except Exception as e:
                st.error(f"Failed to read topic file: {e}")
                return
        elif topics_text.strip():
            topics = parse_topics_from_text(topics_text)

        use_gemini = engine == "gemini"
        model = None
        vec = None
        topic_vecs = None

        if use_gemini:
            api_key = resolve_gemini_api_key(gemini_key_input)
            if not api_key:
                st.error(
                    "Gemini requires an API key. Switch sidebar to **Local — no API key**, or set **GEMINI_API_KEY** / "
                    "secrets / sidebar key. See “Get a free Gemini API key” in the sidebar."
                )
                return
            configure_gemini(api_key)
            model = make_gemini_model(gemini_model)

        if not topics:
            if use_gemini and model is not None:
                st.info("No topics provided — asking Gemini to propose topic labels from sample remarks...")
                topics = gemini_suggest_topics(model, originals, int(topic_count))
            if not topics:
                st.info("Auto-generating topic labels from remarks (TF-IDF)...")
                topics = generate_topics_from_remarks(
                    originals,
                    topic_count=int(topic_count),
                    seed_max_rows=200,
                )

        if not topics:
            topics = ["Other"]

        st.write(f"Using {len(topics)} topics.")

        if not use_gemini:
            vec, topic_vecs = fit_topic_vectorizer(topics)
            st.info("Running **local** pipeline (no API calls): extractive summaries, TF-IDF topic match, VADER sentiment.")

        results_summarized: List[str] = []
        results_topics: List[str] = []
        results_sentiment: List[str] = []

        progress = st.progress(0)
        status = st.empty()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = originals[batch_start:batch_end]
            status.write(f"Processing batch {batch_start // batch_size + 1} / {(total + batch_size - 1) // batch_size} (rows {batch_start+1}-{batch_end})")
            progress.progress(min(1.0, batch_end / total))

            if use_gemini:
                try:
                    summarized, batch_topics, batch_sentiments = gemini_process_batch(model, batch, topics)
                except Exception as e:
                    st.error(f"Gemini batch failed (rows {batch_start + 1}-{batch_end}): {e}")
                    return
            else:
                assert vec is not None and topic_vecs is not None
                summarized, batch_topics, batch_sentiments = local_process_batch(batch, topics, vec, topic_vecs)

            results_summarized.extend(summarized)
            results_topics.extend(batch_topics)
            results_sentiment.extend(batch_sentiments)

        progress.progress(1.0)
        status.write("Processing complete.")

        # Build final output dataset
        out_df = df_clean.copy()
        out_df.drop(columns=[c for c in [remark_col] if c in out_df.columns], inplace=True)
        out_df.rename(columns={"__original_remark": "Original Remark"}, inplace=True)
        out_df["Summarized Remark"] = results_summarized
        out_df["Topic"] = results_topics
        out_df["Sentiment"] = results_sentiment

        st.session_state["output_df"] = out_df
        st.session_state["output_ready"] = True

    if st.session_state.get("output_ready") and "output_df" in st.session_state:
        st.subheader("Preview")
        st.dataframe(st.session_state["output_df"].head(50), use_container_width=True)

        render_output_analysis(st.session_state["output_df"])

        download_bytes = build_download_xlsx(st.session_state["output_df"])
        st.download_button(
            label="Download .xlsx",
            data=download_bytes,
            file_name="processed_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )


if __name__ == "__main__":
    main()

