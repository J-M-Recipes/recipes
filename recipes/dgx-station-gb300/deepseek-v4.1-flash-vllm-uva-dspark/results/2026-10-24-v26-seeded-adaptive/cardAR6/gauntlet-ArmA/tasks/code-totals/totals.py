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
        amount_text = record.get("amount", "").strip()
        if not category or not amount_text:
            continue
        totals[category] = totals.get(category, 0) + int(amount_text)
    return totals
