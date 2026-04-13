"""
========================================================
  Feedback Batch Processor — Claude AI Pipeline
  Author: Generated for Cursor
  Description: Classifies, summarizes, and analyzes
               sentiment on patient/customer feedback
               using the Anthropic API in batches.
========================================================

SETUP:
  pip install anthropic pandas openpyxl

USAGE:
  1. Set your ANTHROPIC_API_KEY environment variable.
  2. Edit CONFIG below (file paths, column names, batch size).
  3. Run: python process_feedback.py
"""

import os
import json
import time
import anthropic
import pandas as pd

# ─────────────────────────────────────────────
#  CONFIG — edit these before running
# ─────────────────────────────────────────────
CONFIG = {
    # Path to your input Excel file
    "input_file": "D:\Practice\Project1\SentimentAnalysis\patient_feedback_1000_rows_detailed.csv",

    # Path to your topic-dictionary Excel/CSV file.
    # The file must have a column named "Topic" listing all valid labels.
    # Set to None to pass topics inline (see TOPIC_LIST below).
    "topic_dict_file": "D:\Practice\Project1\SentimentAnalysis\Topics.csv",

    # If topic_dict_file is None, list your topics here:
    #"topic_list_fallback": [
     #   "Billing & Payments",
      #  "Staff Behavior",
       # "Wait Time",
        #"Cleanliness",
        "Treatment Quality",
        "Appointment Scheduling",
        "Communication",
        "Facility & Equipment",
        "Food & Amenities",
        "Discharge Process",
        "Other",
    ],

    # Column names in your input Excel file
    "id_column": "Response_ID",       # Unique row identifier
    "comment_column": "Comments",     # Column that holds the raw comments

    # Processing settings
    "batch_size": 20,                 # Comments per API call
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 4000,

    # Output
    "output_file": "Processed_Output.xlsx",
    "checkpoint_file": "checkpoint.json",  # Saves progress so you can resume
}
# ─────────────────────────────────────────────


def load_topics(cfg: dict) -> list[str]:
    """Load the topic list from a file or fall back to the inline list."""
    if cfg["topic_dict_file"] and os.path.exists(cfg["topic_dict_file"]):
        ext = os.path.splitext(cfg["topic_dict_file"])[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(cfg["topic_dict_file"])
        else:
            df = pd.read_csv(cfg["topic_dict_file"])
        # Accept either "Topic" or "topic" column
        col = next((c for c in df.columns if c.lower() == "topic"), None)
        if col is None:
            raise ValueError(
                f"Topic dictionary file must contain a 'Topic' column. "
                f"Found columns: {list(df.columns)}"
            )
        topics = df[col].dropna().unique().tolist()
        print(f"✅ Loaded {len(topics)} topics from '{cfg['topic_dict_file']}'")
        return topics
    print("ℹ️  Using inline topic list from CONFIG.")
    return cfg["topic_list_fallback"]


def load_data(cfg: dict) -> pd.DataFrame:
    """Load the input Excel file and drop rows with empty comments."""
    df = pd.read_excel(cfg["input_file"])
    original_len = len(df)
    df = df.dropna(subset=[cfg["comment_column"]])
    df = df[df[cfg["comment_column"]].astype(str).str.strip() != ""]
    print(
        f"✅ Loaded '{cfg['input_file']}': "
        f"{original_len} rows → {len(df)} valid rows after dropping blanks."
    )
    return df.reset_index(drop=True)


def load_checkpoint(path: str) -> dict:
    """Load previously saved batch results so we can resume interrupted runs."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"♻️  Checkpoint found — resuming from batch {data.get('next_batch', 1)}.")
        return data
    return {"results": [], "next_batch": 1}


def save_checkpoint(path: str, results: list, next_batch: int):
    """Persist progress after each batch."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "next_batch": next_batch}, f, ensure_ascii=False)


def build_prompt(batch_rows: list[dict], topics: list[str]) -> str:
    """
    Construct the prompt for a single batch.
    Returns a prompt that asks Claude to output strict JSON.
    """
    topic_list_str = "\n".join(f"  - {t}" for t in topics)
    comments_block = "\n".join(
        f'{i + 1}. [ID: {row["id"]}] {row["comment"]}'
        for i, row in enumerate(batch_rows)
    )

    return f"""You are a data-processing assistant. Process each comment below and return ONLY a valid JSON array — no preamble, no markdown fences, no extra text.

ALLOWED TOPICS (use exactly as written, choose the single best match):
{topic_list_str}

INSTRUCTIONS PER COMMENT:
1. Summarization: Write a concise English summary (1-2 sentences). If the comment is not in English, translate first, then summarize.
2. Topic: Pick ONE label from the allowed list above. Do not invent new labels.
3. Sentiment: One of "Positive", "Negative", or "Neutral".

COMMENTS TO PROCESS:
{comments_block}

REQUIRED OUTPUT FORMAT — a JSON array with one object per comment:
[
  {{
    "id": "<original Response_ID>",
    "summarization": "<summary>",
    "topic": "<topic label>",
    "sentiment": "<Positive|Negative|Neutral>"
  }},
  ...
]

Return ONLY the JSON array. Nothing else."""


def call_claude(client: anthropic.Anthropic, prompt: str, cfg: dict) -> list[dict]:
    """Send one batch to the Claude API and parse the JSON response."""
    message = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


def process_batches(
    df: pd.DataFrame,
    topics: list[str],
    client: anthropic.Anthropic,
    cfg: dict,
) -> list[dict]:
    """Split data into batches, call Claude, and collect results."""
    checkpoint = load_checkpoint(cfg["checkpoint_file"])
    results: list[dict] = checkpoint["results"]
    next_batch: int = checkpoint["next_batch"]

    id_col = cfg["id_column"]
    comment_col = cfg["comment_column"]
    batch_size = cfg["batch_size"]

    rows = [
        {"id": str(row[id_col]), "comment": str(row[comment_col])}
        for _, row in df.iterrows()
    ]
    total_batches = (len(rows) + batch_size - 1) // batch_size

    for batch_num in range(next_batch, total_batches + 1):
        start = (batch_num - 1) * batch_size
        end = start + batch_size
        batch_rows = rows[start:end]

        print(f"  ⏳ Batch {batch_num}/{total_batches} ({len(batch_rows)} comments) ...", end=" ", flush=True)
        t0 = time.time()

        prompt = build_prompt(batch_rows, topics)

        # Retry up to 3 times on transient errors
        for attempt in range(1, 4):
            try:
                batch_results = call_claude(client, prompt, cfg)
                break
            except (json.JSONDecodeError, Exception) as exc:
                if attempt == 3:
                    print(f"\n  ❌ Batch {batch_num} failed after 3 attempts: {exc}")
                    # Fill with error placeholders so we don't lose row alignment
                    batch_results = [
                        {
                            "id": r["id"],
                            "summarization": "ERROR",
                            "topic": "Other",
                            "sentiment": "Neutral",
                        }
                        for r in batch_rows
                    ]
                else:
                    wait = 5 * attempt
                    print(f"\n  ⚠️  Attempt {attempt} failed ({exc}). Retrying in {wait}s ...")
                    time.sleep(wait)

        results.extend(batch_results)
        save_checkpoint(cfg["checkpoint_file"], results, batch_num + 1)
        print(f"done in {time.time() - t0:.1f}s")

    return results


def merge_and_export(df: pd.DataFrame, results: list[dict], cfg: dict):
    """Merge AI results back into the original DataFrame and export to Excel."""
    id_col = cfg["id_column"]

    # Build a lookup by ID (string-keyed for safety)
    lookup = {str(r["id"]): r for r in results}

    df = df.copy()
    df["Summarization"] = df[id_col].astype(str).map(
        lambda x: lookup.get(x, {}).get("summarization", "")
    )
    df["Topic"] = df[id_col].astype(str).map(
        lambda x: lookup.get(x, {}).get("topic", "")
    )
    df["Sentiment"] = df[id_col].astype(str).map(
        lambda x: lookup.get(x, {}).get("sentiment", "")
    )

    df.to_excel(cfg["output_file"], index=False)
    print(f"\n✅ Final output saved → '{cfg['output_file']}' ({len(df)} rows)")


def main():
    print("=" * 55)
    print("  Feedback Batch Processor — Claude AI Pipeline")
    print("=" * 55)

    # Validate API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is not set.\n"
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client = anthropic.Anthropic(api_key=api_key)

    # Load inputs
    topics = load_topics(CONFIG)
    df = load_data(CONFIG)

    print(f"\n🔄 Processing {len(df)} comments in batches of {CONFIG['batch_size']}...")
    results = process_batches(df, topics, client, CONFIG)

    print(f"\n🔗 Merging {len(results)} results into original dataset...")
    merge_and_export(df, results, CONFIG)

    # Clean up checkpoint file after successful run
    if os.path.exists(CONFIG["checkpoint_file"]):
        os.remove(CONFIG["checkpoint_file"])
        print("🧹 Checkpoint file removed.")

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()