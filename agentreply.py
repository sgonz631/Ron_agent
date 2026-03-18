import ronnor_inventory as inv

def handle_customer_request(search_text, size):
    tag = inv.normalize_tag(search_text)

    result = inv.search_anything(search_text, size, tag)

    response_parts = [result]

    if "don't see" in result or "don’t see" in result:
        closest = inv.suggest_closest_size(tag, size)
        if closest:
            response_parts.append(closest)

    recs = inv.recommend_shoes(tag)
    if recs:
        response_parts.append(recs)

    promos = inv.show_promotions()
    if promos:
        response_parts.append(promos)

    return "\n\n".join(response_parts)