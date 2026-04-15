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
    return text.lower().strip() if text else ""


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
        "size", "brand", "model", "color", "promotion", "promotions",
        "deal", "deals"
    ]

    if any(word in text for word in inventory_words):
        return True
    if extract_brand(text):
        return True
    if extract_tags(text):
        return True

    return False


def parse_inventory_request(user_text: str) -> dict:
    text = normalize_text(user_text)
    return {
        "brand": extract_brand(text),
        "size": extract_size(text),
        "color": extract_color(text),
        "tags": extract_tags(text),
        "wants_promotions": any(
            word in text for word in ["promotion", "promotions", "deal", "deals", "discount"]
        ),
    }


def get_inventory_filters(user_text: str):
    """
    Parse inventory-like user text into filters only.
    Returns None if this does not look like an inventory request.
    """
    if not seems_inventory_request(user_text):
        return None

    return parse_inventory_request(user_text)


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


def rank_inventory_rows(rows: list, filters: dict) -> list:
    """
    Rank rows so the best matches appear first.
    More matching tags, requested promotions, and higher stock score better.
    """
    desired_tags = set(filters.get("tags", []))
    wants_promotions = filters.get("wants_promotions", False)

    def score(row):
        brand, model, size, color, cost, qty, location, tags, promotion = row
        tag_set = set()

        if tags:
            tag_set = {t.strip().lower() for t in tags.split(",") if t.strip()}

        score_value = 0
        score_value += len(desired_tags.intersection(tag_set)) * 10

        if wants_promotions and promotion and promotion.strip():
            score_value += 8

        score_value += min(qty, 10)
        return score_value

    return sorted(rows, key=score, reverse=True)


def build_inventory_context(user_text: str, filters: dict, rows: list) -> str:
    """
    Build a compact, factual context block for Ollama.
    """
    if not rows:
        return (
            f"User request: {user_text}\n"
            f"Parsed filters: {filters}\n"
            "Inventory results: none\n"
        )

    lines = [
        f"User request: {user_text}",
        f"Parsed filters: {filters}",
        f"Inventory result count: {len(rows)}",
        "Inventory results:"
    ]

    for row in rows[:8]:
        brand, model, size, color, cost, qty, location, tags, promotion = row
        line = (
            f"- {brand} {model} | size {size} | color {color} | "
            f"price ${cost:.0f} | qty {qty}"
        )
        if location:
            line += f" | location {location}"
        if tags:
            line += f" | tags {tags}"
        if promotion:
            line += f" | promotion {promotion}"
        lines.append(line)

    if len(rows) > 8:
        lines.append(f"- plus {len(rows) - 8} more matching result(s)")

    return "\n".join(lines)


def get_inventory_context(user_text: str):
    """
    Full helper: parse, search, rank, and build context.
    Kept for compatibility, but chatbot.py can use get_inventory_filters()
    to avoid duplicate searches after merging session preferences.
    """
    filters = get_inventory_filters(user_text)
    if not filters:
        return None

    rows = rank_inventory_rows(search_inventory(filters), filters)
    context = build_inventory_context(user_text, filters, rows)

    return {
        "filters": filters,
        "rows": rows,
        "context": context,
    }
