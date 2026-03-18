import sqlite3

DB_PATH = "shoe_store.db"

TAG_SYNONYMS = {
    "sport": "sportwear",
    "sports": "sportwear",
    "gym": "sportwear",
    "athletic": "sportwear",
    "workout": "sportwear",

    "running": "running",
    "runner": "running",

    "casual": "casual",
    "everyday": "casual",

    "street": "streetwear",
    "streetwear": "streetwear",

    "classic": "classic",
    "lifestyle": "lifestyle"
}

def normalize_tag(text):
    words = text.lower().strip().split()

    for word in words:
        if word in TAG_SYNONYMS:
            return TAG_SYNONYMS[word]

    return words[0] if words else text

def query_db(sql, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()

def search_anything(search_text, size, tag, color=None):
    sql = """
SELECT brand, model, primary_color, cost, quantity
FROM shoes
WHERE size = ?
  AND quantity > 0
  AND (? IS NULL OR primary_color = ?)
  AND (
    brand LIKE '%' || ? || '%'
    OR model LIKE '%' || ? || '%'
    OR tags LIKE '%' || ? || '%'
)
ORDER BY brand, model;
"""

    rows = query_db(sql, (size, color, color, search_text, search_text, tag))


    if not rows:
        return f"Hmm, I don't see '{search_text}' in size {size} right now."

    response = ["Nice! Here's what I found:"]
    for b, m, c, cost, q in rows:
        response.append(f"- {b} {m} ({c}) — ${cost:.2f} (qty {q})")
    return "\n".join(response)

def recommend_shoes(tag):
    sql = """
    SELECT brand, model, primary_color, cost
    FROM shoes
    WHERE tags LIKE '%' || ? || '%'
    AND quantity > 0
    ORDER BY RANDOM()
    LIMIT 2;
    """

    rows = query_db(sql, (tag,))

    if not rows:
        return "I don't have other similar options right now, but I can help you find something else if you want!"

    response = ["You might like these too:"]
    for b, m, c, cost in rows:
        response.append(f"- {b} {m} ({c}) — ${cost:.2f}")

    return "\n".join(response)

def show_promotions():
    sql = """
    SELECT brand, model, primary_color, promotion
    FROM shoes
    WHERE promotion IS NOT NULL
    AND promotion != ''
    ORDER BY brand, model;
    """

    rows = query_db(sql)

    if not rows:
        return "There are currently no promotions."

    response = ["Here are a few deals going on right now:"]
    for b, m, c, promo in rows:
        response.append(f"- {b} {m} ({c}) — {promo}")

    return "\n".join(response)

def suggest_closest_size(tag, requested_size):
    sql = """
    SELECT brand, model, primary_color, size, cost
    FROM shoes
    WHERE tags LIKE '%' || ? || '%'
    AND quantity > 0
    ORDER BY ABS(size - ?) ASC
    LIMIT 1;
    """

    rows = query_db(sql, (tag, requested_size))

    if not rows:
        return None

    b, m, c, s, cost = rows[0]
    return f"I don’t have your exact size, but the {b} {m} ({c}) is available in size {s} for ${cost:.2f}."

def suggest_similar(tag):
    sql = """
    SELECT brand, model, primary_color, cost
    FROM shoes
    WHERE tags LIKE '%' || ? || '%'
    AND quantity > 0
    ORDER BY RANDOM()
    LIMIT 2;
    """

    rows = query_db(sql, (tag,))

    if not rows:
        return "I couldn't find similar items right now."

    response = ["However, you might like:"]
    for b, m, c, cost in rows:
        response.append(f"- {b} {m} ({c}) — ${cost:.2f}")

    return "\n".join(response)

def greet_customer():
    return "Hey! I'm Ron. Looking for a specific shoe or just browsing today?"

# ---- Simulated interaction ----
if __name__ == "__main__":
    print(greet_customer())

    search_text = input("What shoe are you looking for? (brand/model/style): ")
    size = float(input("What size?: "))

    tag = normalize_tag(search_text)

    result = search_anything(search_text, size, tag)

    print(result)

    if "Hmm, I don't see" in result:
        closest = suggest_closest_size(tag, size)
        if closest:
            print(closest)

    print(recommend_shoes(tag))
    print(show_promotions())