import sqlite3
import re
from pathlib import Path

DB_PATH = Path("shoe_store.db")

KNOWN_BRANDS = [
    "nike", "adidas", "puma", "reebok", "asics",
    "new balance", "converse", "vans", "under armour"
]

KNOWN_COLORS = [
    "black", "white", "red", "blue", "green", "gray", "grey",
    "pink", "brown", "navy", "olive", "silver"
]

TAG_SYNONYMS = {
    "running": ["running", "runner", "jogging"],
    "training": ["training", "gym", "workout", "cross training", "cross-training"],
    "walking": ["walking", "walk"],
    "casual": ["casual"],
    "everyday": ["everyday", "daily"],
    "lifestyle": ["lifestyle"],
    "streetwear": ["streetwear"],
    "classic": ["classic"],
    "retro": ["retro", "vintage"],
    "minimal": ["minimal", "simple"],
    "bold": ["bold"],
    "platform": ["platform"],
    "comfort": ["comfort", "comfortable", "comfy"],
    "lightweight": ["lightweight", "light"],
    "breathable": ["breathable", "airy", "ventilated"],
    "durable": ["durable", "tough"],
    "support": ["support", "supportive", "arch support", "arch-support"],
    "cushioned": ["cushioned", "cushioning", "soft"],
    "waterproof": ["waterproof", "water resistant", "water-resistant"],
    "wide": ["wide"],
    "narrow": ["narrow"],
    "hiking": ["hiking", "trail", "outdoor"],
    "sporty": ["sporty"],
    "sportswear": ["sportswear", "athletic"],
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.lower().strip()


def extract_size(user_text: str):
    text = normalize_text(user_text)

    match = re.search(r"\bsize\s*(\d+(?:\.\d+)?)\b", text)
    if match:
        return float(match.group(1))

    match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if match:
        try:
            value = float(match.group(1))
            if 4 <= value <= 18:
                return value
        except ValueError:
            pass

    return None


def extract_brand(user_text: str) -> str:
    text = normalize_text(user_text)
    for brand in KNOWN_BRANDS:
        if brand in text:
            return brand
    return ""


def extract_color(user_text: str) -> str:
    text = normalize_text(user_text)
    for color in KNOWN_COLORS:
        if color in text:
            return color
    return ""


def extract_tags(user_text: str) -> list[str]:
    text = normalize_text(user_text)
    found = []

    for canonical, words in TAG_SYNONYMS.items():
        if canonical in text:
            found.append(canonical)
            continue

        for word in words:
            if word in text:
                found.append(canonical)
                break

    deduped = []
    for tag in found:
        if tag not in deduped:
            deduped.append(tag)

    return deduped


def seems_inventory_request(user_text: str) -> bool:
    text = normalize_text(user_text)

    inventory_words = [
        "shoe", "shoes", "sneaker", "sneakers",
        "inventory", "stock", "have", "carry", "available",
        "size", "brand", "model", "color", "promotion", "promotions"
    ]

    if any(word in text for word in inventory_words):
        return True

    if extract_brand(text):
        return True

    if extract_tags(text):
        return True

    return False


def parse_inventory_request(user_text: str) -> dict:
    return {
        "brand": extract_brand(user_text),
        "size": extract_size(user_text),
        "color": extract_color(user_text),
        "tags": extract_tags(user_text),
        "wants_promotions": "promotion" in user_text.lower() or "promotions" in user_text.lower() or "deal" in user_text.lower(),
    }


def search_inventory(filters: dict) -> list:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH.resolve()}")

    query = """
    SELECT brand, model, size, primary_color, cost, quantity, location, tags, promotion
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

    for tag in filters.get("tags", []):
        query += " AND LOWER(tags) LIKE ?"
        params.append(f"%{tag}%")

    if filters.get("wants_promotions"):
        query += " AND promotion IS NOT NULL AND TRIM(promotion) != ''"

    query += " ORDER BY brand, model, size"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows


def format_inventory_response(rows: list, filters: dict) -> str:
    if not rows:
        return "I could not find any matching shoes in inventory."

    count = len(rows)

    # ---------------------------------------------------
    # SHORT + NATURAL RESPONSE
    # ---------------------------------------------------
    if count == 1:
        brand, model, size, color, cost, qty, location, tags, promotion = rows[0]
        return (
            f"I found one option: {brand} {model}, size {size}, {color}, "
            f"priced at {cost:.0f} dollars. Want more details?"
        )

    # ---------------------------------------------------
    # MULTIPLE RESULTS → summarize instead of dump
    # ---------------------------------------------------
    models = list({f"{r[0]} {r[1]}" for r in rows})

    # pick top 2–3 examples
    examples = models[:3]

    response = f"I found {count} options. "

    response += "Some popular ones are: "
    response += ", ".join(examples) + "."

    # Add context if relevant
    if filters.get("wants_promotions"):
        response += " Some of these are currently on promotion."

    # Add follow-up guidance
    response += " Want me to narrow it down by size, price, or style?"

    return response


def handle_inventory_query(user_text: str) -> str | None:
    if not seems_inventory_request(user_text):
        return None

    filters = parse_inventory_request(user_text)
    rows = search_inventory(filters)
    return format_inventory_response(rows, filters)
