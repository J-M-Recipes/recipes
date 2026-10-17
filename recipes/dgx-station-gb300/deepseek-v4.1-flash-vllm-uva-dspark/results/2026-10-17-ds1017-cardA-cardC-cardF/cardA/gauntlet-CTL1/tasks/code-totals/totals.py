def totals_by_category(csv_text):
    totals = {}
    lines = csv_text.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        if i == 0 and line.lower().startswith("category"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        category = parts[0].strip()
        amount = parts[1].strip()
        if not category or not amount:
            continue
        totals[category] = totals.get(category, 0) + int(amount)
    return totals
