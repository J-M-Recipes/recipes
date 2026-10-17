def totals_by_category(csv_text):
    totals = {}
    header_seen = False
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if not header_seen:
            header_seen = True
            if parts[0].strip().lower() == "category":
                continue
        if len(parts) < 2:
            continue
        category = parts[0].strip()
        amount = int(parts[1].strip())
        totals[category] = totals.get(category, 0) + amount
    return totals
