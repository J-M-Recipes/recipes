import csv
import io


def totals_by_category(csv_text):
    totals = {}
    reader = csv.reader(io.StringIO(csv_text))
    header = None
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if header is None:
            header = [cell.strip() for cell in row]
            continue
        record = dict(zip(header, row))
        category = record.get("category", "").strip()
        amount = int(record.get("amount", "0").strip())
        totals[category] = totals.get(category, 0) + amount
    return totals
