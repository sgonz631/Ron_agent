import time

def set_expression(shared_state, new_state):
    now = time.perf_counter()

    old_state = shared_state.get("expression", "idle")
    last_change = shared_state.get("state_started_at", now)

    duration = now - last_change

    stats = shared_state.setdefault("state_stats", {})
    if old_state not in stats:
        stats[old_state] = {"total": 0.0, "count": 0, "last": 0.0}

    stats[old_state]["total"] += duration
    stats[old_state]["count"] += 1
    stats[old_state]["last"] = duration

    print(f"[STATE] {old_state} lasted {duration:.2f} sec")

    shared_state["expression"] = new_state
    shared_state["state_started_at"] = now


def print_state_summary(shared_state):
    print("\n[STATE SUMMARY]")
    stats = shared_state.get("state_stats", {})
    for state, data in stats.items():
        avg = data["total"] / data["count"] if data["count"] else 0
        print(
            f"  {state:<10} total={data['total']:.2f}s  "
            f"count={data['count']}  avg={avg:.2f}s  last={data['last']:.2f}s"
        )