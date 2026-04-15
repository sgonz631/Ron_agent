import sqlite3
import json
import re
import requests
from pathlib import Path
from typing import Optional

# --------------------------------------------------
# DATABASE PATH
# --------------------------------------------------
DB_PATH = Path("shoe_store.db")

# --------------------------------------------------
# MODEL CONFIG
# --------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_PARSE_MODEL = "gemma3:4b"
OLLAMA_REPLY_MODEL = "gemma3:1b"
OLLAMA_PARSE_TIMEOUT = 90
OLLAMA_REPLY_TIMEOUT = 90

# --------------------------------------------------
# TAG NORMALIZATION
# Maps user words to canonical searchable tags
# --------------------------------------------------
TAG_SYNONYMS = {
    "running": ["running", "runner", "jogging", "cardio"],
    "training": ["training", "gym", "workout", "cross training"],
    "walking": ["walking", "walk", "comfort"],
    "casual": ["casual", "everyday", "daily", "lifestyle"],
    "basketball": ["basketball", "hoops", "court"],
    "skate": ["skate", "skateboarding"],
    "dress": ["dress", "formal", "office"],
    "boots": ["boots", "boot", "outdoor"],
    "sandals": ["sandals", "slides", "flip flops"],
    "streetwear": ["street", "streetwear"],
    "classic": ["classic", "retro"],
    "sportwear": ["sport", "sports", "athletic", "sportwear"],
}

IGNORE_TAG_WORDS = {
    "shoe", "shoes", "sneaker", "sneakers", "footwear", "pair"
}

KNOWN_BRANDS = sorted([
    "under armour",
    "new balance",
    "nike",
    "adidas",
    "puma",
    "reebok",
    "asics",
    "converse",
    "vans",
], key=len, reverse=True)

KNOWN_COLORS = [
    "black", "white", "red", "blue", "green",
    "gray", "grey", "pink", "brown", "navy"
]

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def clean_text(text: str) -> str:
    """Basic cleanup for user input."""
    return re.sub(r"\s+", " ", text.strip())


def safe_print(text: str) -> None:
    """Print safely even if terminal encoding is limited."""
    try:
        print(text)
    except UnicodeEncodeError:
        fallback = text.encode("ascii", errors="replace").decode("ascii")
        print(fallback)


def normalize_style(style_text: str) -> str:
    """Convert style-like text into a canonical searchable tag."""
    if not style_text:
        return ""

    style_text = style_text.lower().strip()

    if style_text in IGNORE_TAG_WORDS:
        return ""

    for canonical, words in TAG_SYNONYMS.items():
        if style_text == canonical:
            return canonical
        for word in words:
            if word in style_text:
                return canonical

    return ""


def extract_tags(text: str) -> list[str]:
    """Extract multiple canonical tags from user text."""
    text = text.lower()
    found: list[str] = []

    for canonical, words in TAG_SYNONYMS.items():
        if canonical in IGNORE_TAG_WORDS:
            continue

        if canonical in text:
            found.append(canonical)
            continue

        for word in words:
            if word in IGNORE_TAG_WORDS:
                continue
            if word in text:
                found.append(canonical)
                break

    return list(dict.fromkeys(found))


def empty_filters() -> dict:
    """Return a clean filter state."""
    return {
        "action": "search_inventory",
        "activity": "",
        "style": "",
        "brand": "",
        "size": None,
        "color": "",
        "max_price": None,
        "tags": [],
    }


def merge_filters(current: dict, new_data: dict) -> dict:
    """Merge newly extracted filters into saved conversation filters."""
    merged = current.copy()

    for key, value in new_data.items():
        if key not in merged:
            continue

        if key == "tags":
            current_tags = merged.get("tags", [])
            new_tags = value if isinstance(value, list) else []
            merged["tags"] = list(dict.fromkeys(current_tags + new_tags))
            continue

        if value not in ("", None):
            merged[key] = value

    return merged


# --------------------------------------------------
# DIRECT FIELD EXTRACTION
# Used when Ron has just asked for one specific field
# --------------------------------------------------
def extract_for_expected_field(user_text: str, expected_field: Optional[str]) -> dict:
    text = clean_text(user_text).lower()

    if expected_field == "brand":
        for brand in KNOWN_BRANDS:
            if brand in text:
                return {"brand": brand}
        return {"brand": text} if text else {}

    if expected_field == "size":
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            return {"size": float(match.group(1))}
        return {}

    if expected_field == "max_price":
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            return {"max_price": float(match.group(1))}
        return {}

    if expected_field == "activity":
        tags = extract_tags(text)
        value = tags[0] if tags else text
        return {
            "activity": value,
            "style": value if value in TAG_SYNONYMS else "",
            "tags": tags,
        }

    return {}


# --------------------------------------------------
# FALLBACK PARSER
# Used if Ollama fails or times out
# --------------------------------------------------
def fallback_extract(user_text: str) -> dict:
    text = user_text.lower()

    size = None
    size_match = re.search(r"\b(?:size\s*)?(\d+(?:\.\d+)?)\b", text)
    if size_match:
        size = float(size_match.group(1))

    max_price = None
    price_match = re.search(r"(?:under|below|max|maximum|budget)\s*\$?(\d+(?:\.\d+)?)", text)
    if price_match:
        max_price = float(price_match.group(1))

    brand = ""
    for b in KNOWN_BRANDS:
        if b in text:
            brand = b
            break

    style = ""
    for canonical, words in TAG_SYNONYMS.items():
        if canonical in text:
            style = canonical
            break
        for word in words:
            if word in text:
                style = canonical
                break
        if style:
            break

    activity = style
    color = ""

    for c in KNOWN_COLORS:
        if c in text:
            color = c
            break

    return {
        "action": "search_inventory",
        "activity": activity,
        "style": style,
        "brand": brand,
        "size": size,
        "color": color,
        "max_price": max_price,
        "tags": extract_tags(user_text),
    }


# --------------------------------------------------
# OLLAMA PARSER
# Uses Ollama to extract structured filters from user text
# Falls back to local parsing if needed
# --------------------------------------------------
def parse_with_ollama(user_text: str) -> dict:
    prompt = f"""Return ONLY valid JSON.
Do not explain anything.
Use exactly this schema:

{{
  "action": "search_inventory",
  "activity": "",
  "style": "",
  "brand": "",
  "size": null,
  "color": "",
  "max_price": null,
  "tags": []
}}

User request: {user_text}
"""

    try:
        print("[DEBUG] Sending request to Ollama HTTP API...", flush=True)

        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_PARSE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=OLLAMA_PARSE_TIMEOUT,
        )
        response.raise_for_status()

        raw = response.json()["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        print(f"[DEBUG] Raw output:\n{raw}\n", flush=True)

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in Ollama output.")

        parsed = json.loads(raw[start:end + 1])

        parsed.setdefault("action", "search_inventory")
        parsed.setdefault("activity", "")
        parsed.setdefault("style", "")
        parsed.setdefault("brand", "")
        parsed.setdefault("size", None)
        parsed.setdefault("color", "")
        parsed.setdefault("max_price", None)
        parsed.setdefault("tags", [])

        parsed["activity"] = str(parsed["activity"]).lower().strip() if parsed["activity"] else ""
        parsed["style"] = normalize_style(str(parsed["style"])) if parsed["style"] else ""
        if parsed["style"] in IGNORE_TAG_WORDS:
            parsed["style"] = ""
        parsed["brand"] = str(parsed["brand"]).lower().strip() if parsed["brand"] else ""
        parsed["color"] = str(parsed["color"]).lower().strip() if parsed["color"] else ""

        if parsed["size"] not in (None, ""):
            try:
                parsed["size"] = float(parsed["size"])
            except (ValueError, TypeError):
                parsed["size"] = None

        if parsed["max_price"] not in (None, ""):
            try:
                parsed["max_price"] = float(parsed["max_price"])
            except (ValueError, TypeError):
                parsed["max_price"] = None

        parsed["tags"] = extract_tags(user_text)

        if not parsed["style"] and parsed["tags"]:
            parsed["style"] = parsed["tags"][0]

        if not parsed["activity"] and parsed["style"]:
            parsed["activity"] = parsed["style"]

        return parsed

    except Exception as e:
        print(f"[Ollama parse failed, using fallback] {e}", flush=True)
        return fallback_extract(user_text)


# --------------------------------------------------
# DATABASE SEARCH
# Exact search using all current filters
# --------------------------------------------------
def search_inventory(filters: dict) -> list:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH.resolve()}")

    query = """
    SELECT brand, model, size, primary_color, cost, quantity, tags
    FROM shoes
    WHERE quantity > 0
    """
    params = []

    if filters.get("brand"):
        query += " AND LOWER(brand) LIKE ?"
        params.append(f"%{filters['brand']}%")

    if filters.get("size") is not None:
        query += " AND size = ?"
        params.append(filters["size"])

    if filters.get("color"):
        query += " AND LOWER(primary_color) LIKE ?"
        params.append(f"%{filters['color']}%")

    tag_terms = []

    if filters.get("style"):
        tag_terms.append(filters["style"])

    if filters.get("tags"):
        for tag in filters["tags"]:
            if tag not in tag_terms:
                tag_terms.append(tag)

    for tag in tag_terms:
        query += " AND LOWER(tags) LIKE ?"
        params.append(f"%{tag}%")

    if filters.get("max_price") is not None:
        query += " AND cost <= ?"
        params.append(filters["max_price"])

    print(f"[DEBUG] SQL: {query}", flush=True)
    print(f"[DEBUG] PARAMS: {params}", flush=True)

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()


# --------------------------------------------------
# RELAXED SEARCH
# Safe fallback search
# --------------------------------------------------
def search_inventory_relaxed(filters: dict) -> tuple[list, str]:
    rows = search_inventory(filters)
    if rows:
        return rows, "exact"

    if filters.get("size") is not None:
        relaxed = filters.copy()
        relaxed["size"] = None
        print("[DEBUG] No exact matches. Retrying without size...", flush=True)
        rows = search_inventory(relaxed)
        if rows:
            return rows, "without_size"

    if filters.get("color"):
        relaxed = filters.copy()
        relaxed["color"] = ""
        print("[DEBUG] No exact matches. Retrying without color...", flush=True)
        rows = search_inventory(relaxed)
        if rows:
            return rows, "without_color"

    if filters.get("style") or filters.get("tags"):
        relaxed = filters.copy()
        relaxed["style"] = ""
        relaxed["tags"] = []
        print("[DEBUG] No exact matches. Retrying without style/tags...", flush=True)
        rows = search_inventory(relaxed)
        if rows:
            return rows, "without_style"

    return [], "none"


# --------------------------------------------------
# OLLAMA FINAL RESPONSE GENERATION
# --------------------------------------------------
def rows_to_text(rows: list) -> str:
    if not rows:
        return "No matching products found."

    lines = []
    for brand, model, size, color, cost, qty, tags in rows:
        lines.append(
            f"- {brand} {model}, size {size}, color {color}, "
            f"price ${cost:.2f}, quantity {qty}, tags: {tags}"
        )
    return "\n".join(lines)


def generate_natural_response(user_request: str, filters: dict, rows: list, match_type: str) -> str:
    prompt = f"""You are Ron, a shoe store assistant.

User request: {user_request}

Filters:
- brand: {filters.get("brand")}
- size: {filters.get("size")}
- color: {filters.get("color")}
- max price: {filters.get("max_price")}
- tags: {filters.get("tags")}

Match type: {match_type}

Results:
{rows_to_text(rows)}

Rules:
- Be short.
- Use plain ASCII only.
- Do not invent products.
- Do not invent measurements or units.
- max_price means at or below that price.
- If no results, clearly say no matches.
- If relaxed search was used, say they are close matches.
"""

    try:
        print("[DEBUG] Generating natural response...", flush=True)

        response = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_REPLY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=OLLAMA_REPLY_TIMEOUT,
        )
        response.raise_for_status()

        print("[DEBUG] Natural response received.", flush=True)
        return response.json()["message"]["content"].strip()

    except Exception as e:
        print(f"[DEBUG] Natural response fallback used: {e}", flush=True)

        if rows:
            brand, model, size, color, cost, qty, tags = rows[0]
            return (
                f"I found {len(rows)} match{'es' if len(rows) != 1 else ''}. "
                f"One option is {brand} {model} in {color}, size {size}, for ${cost:.2f}."
            )

        return "I couldn't find any matching shoes in the inventory."


# --------------------------------------------------
# FOLLOW-UP QUESTION LOGIC
# --------------------------------------------------
def next_missing_field(filters: dict) -> Optional[str]:
    if not filters.get("activity") and not filters.get("style") and not filters.get("tags"):
        return "activity"

    if not filters.get("brand"):
        return "brand"

    if filters.get("size") is None:
        return "size"

    if filters.get("max_price") is None:
        return "max_price"

    return None


def question_for_field(field: str) -> str:
    questions = {
        "activity": "What activity are the shoes for?",
        "brand": "Do you have a preferred brand?",
        "size": "What size do you need?",
        "max_price": "What is your budget or maximum price?",
    }
    return questions.get(field, "Can you tell me more?")


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
def main():
    print("RONNOR Inventory Terminal Test")
    print("Type 'exit' to quit.\n")

    filters = empty_filters()
    conversation_history = []
    pending_field: Optional[str] = None

    print("Ron: Hi! I can help you find shoes.")

    while True:
        user_text = clean_text(input("You: "))

        if user_text.lower() == "exit":
            print("Ron: Goodbye.")
            break

        if not user_text:
            print("Ron: Please type a response.")
            continue

        conversation_history.append({"role": "user", "content": user_text})

        if pending_field:
            extracted = extract_for_expected_field(user_text, pending_field)
            pending_field = None

            if not extracted:
                extracted = parse_with_ollama(user_text)
        else:
            extracted = parse_with_ollama(user_text)

        filters = merge_filters(filters, extracted)

        print("\n[DEBUG] Current filters:")
        print(json.dumps(filters, indent=2))

        missing = next_missing_field(filters)

        if missing:
            pending_field = missing
            question = question_for_field(missing)
            safe_print(f"Ron: {question}")
            conversation_history.append({"role": "assistant", "content": question})
            continue

        try:
            rows, match_type = search_inventory_relaxed(filters)

            reply = generate_natural_response(
                user_request=" ".join(
                    msg["content"] for msg in conversation_history if msg["role"] == "user"
                ),
                filters=filters,
                rows=rows,
                match_type=match_type,
            )

            safe_print(f"\nRon: {reply}")

            if rows:
                filters = empty_filters()
                conversation_history.clear()
                pending_field = None
                print("\nRon: Want to search for another pair?")
            else:
                pending_field = None
                safe_print("Ron: I can check similar options if you want, maybe another color, brand, or budget.")

        except Exception as e:
            print(f"Ron: Error: {e}")
            filters = empty_filters()
            conversation_history.clear()
            pending_field = None


# --------------------------------------------------
# SCRIPT ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()