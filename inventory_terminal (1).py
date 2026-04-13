import sqlite3
import json
import subprocess
import re
from pathlib import Path

DB_PATH = Path("shoe_store.db")

STYLE_SYNONYMS = {
    "running": ["running", "runner", "jogging"],
    "training": ["training", "gym", "workout", "cross training"],
    "walking": ["walking", "walk"],
    "casual": ["casual", "everyday", "daily", "lifestyle"],
    "basketball": ["basketball", "hoops"],
    "skate": ["skate", "skateboarding"],
    "dress": ["dress", "formal"],
    "boots": ["boots", "boot"],
    "sandals": ["sandals", "slides", "flip flops"]
}

KNOWN_BRANDS = [
    "nike", "adidas", "puma", "reebok", "asics",
    "new balance", "converse", "vans", "under armour"
]


def normalize_style(style_text: str) -> str:
    if not style_text:
        return ""
    style_text = style_text.lower().strip()
    for canonical, words in STYLE_SYNONYMS.items():
        if style_text == canonical:
            return canonical
        for word in words:
            if word in style_text:
                return canonical
    return style_text


def fallback_extract(user_text: str) -> dict:
    text = user_text.lower()

    size = None
    size_match = re.search(r'\bsize\s*(\d+(?:\.\d+)?)\b', text)
    if size_match:
        size = float(size_match.group(1))

    brand = ""
    for b in KNOWN_BRANDS:
        if b in text:
            brand = b
            break

    style = ""
    for canonical, words in STYLE_SYNONYMS.items():
        if canonical in text:
            style = canonical
            break
        for word in words:
            if word in text:
                style = canonical
                break
        if style:
            break

    colors = ["black", "white", "red", "blue", "green", "gray", "grey", "pink", "brown", "navy"]
    color = ""
    for c in colors:
        if c in text:
            color = c
            break

    return {
        "action": "search_inventory",
        "style": style,
        "brand": brand,
        "size": size,
        "color": color
    }


def parse_with_ollama(user_text: str) -> dict:
    prompt = f"""
You are a shoe store inventory parser.
Extract the user request into valid JSON only.

Allowed action:
- search_inventory

Return exactly this format:
{{
  "action": "search_inventory",
  "style": "",
  "brand": "",
  "size": null,
  "color": ""
}}

User request: {user_text}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma3:4b", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        raw = result.stdout.strip()

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in Ollama output.")

        parsed = json.loads(raw[start:end + 1])

        parsed.setdefault("action", "search_inventory")
        parsed.setdefault("style", "")
        parsed.setdefault("brand", "")
        parsed.setdefault("size", None)
        parsed.setdefault("color", "")

        parsed["style"] = normalize_style(parsed["style"])
        if parsed["brand"]:
            parsed["brand"] = parsed["brand"].lower().strip()
        if parsed["color"]:
            parsed["color"] = parsed["color"].lower().strip()

        return parsed

    except Exception as e:
        print(f"[Ollama parse failed, using fallback] {e}")
        return fallback_extract(user_text)


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
        query += " AND LOWER(brand) = ?"
        params.append(filters["brand"])

    if filters.get("size") is not None:
        query += " AND size = ?"
        params.append(filters["size"])

    if filters.get("color"):
        query += " AND LOWER(primary_color) LIKE ?"
        params.append(f"%{filters['color']}%")

    if filters.get("style"):
        query += " AND LOWER(tags) LIKE ?"
        params.append(f"%{filters['style']}%")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def print_results(rows: list) -> None:
    if not rows:
        print("\nNo matches found.")
        return

    print(f"\nFound {len(rows)} match(es):\n")
    for i, row in enumerate(rows, start=1):
        brand, model, size, color, cost, qty, tags = row
        print(f"{i}. {brand} {model} | Size {size} | {color} | ${cost} | Qty: {qty} | Tags: {tags}")


def main():
    print("RONNOR Inventory Terminal Test")
    print("Type 'exit' to quit.\n")

    while True:
        user_text = input("Search request: ").strip()
        if user_text.lower() == "exit":
            print("Goodbye.")
            break

        filters = parse_with_ollama(user_text)
        print("\nParsed filters:")
        print(json.dumps(filters, indent=2))

        try:
            rows = search_inventory(filters)
            print_results(rows)
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
