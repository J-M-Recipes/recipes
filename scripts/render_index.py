#!/usr/bin/env python3
"""render_index.py — regenerate the recipe index in README.md from every recipes/*/*/recipe.yaml.

Usage:
    python3 scripts/render_index.py           # rewrite README.md between the index markers
    python3 scripts/render_index.py --check   # exit 1 if README.md is out of date (CI)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- recipe-index:start -->", "<!-- recipe-index:end -->"
STATUS_ICON = {"verified": "✅ verified", "experimental": "🧪 experimental", "draft": "📝 draft", "deprecated": "⛔ deprecated"}


def headline_metric(r: dict) -> str:
    ms = r.get("metrics") or []
    if not ms:
        return "—"
    m = ms[0]
    label = f" *({m['label']})*" if m.get("label") else ""
    return f"{m['value']:g} {m['unit']} @ {m['workload']}{label}"


def render() -> str:
    rows = []
    for p in sorted(ROOT.glob("recipes/*/*/recipe.yaml")):
        r = yaml.safe_load(p.read_text())
        hw, slug = p.parent.parts[-2:]
        rel = p.parent.relative_to(ROOT).as_posix()
        rows.append((hw, slug, r))
    if not rows:
        return "\n_No recipes yet._\n"
    out = ["", "| Hardware | Recipe | Model | Status | Headline | Updated |", "|---|---|---|---|---|---|"]
    for hw, slug, r in rows:
        rel = f"recipes/{hw}/{slug}"
        out.append(
            f"| `{hw}` | [{r['title']}]({rel}/) | {r['model']['name']} ({r['model'].get('quantization', '')}) "
            f"| {STATUS_ICON.get(r['status'], r['status'])} | {headline_metric(r)} | {r['updated']} |"
        )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        print(f"README.md lacks {START} / {END} markers")
        return 1
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render()}\n{END}{tail}"
    if a.check:
        if new != text:
            print("README.md index is stale — run scripts/render_index.py")
            return 1
        print("README.md index up to date")
        return 0
    readme.write_text(new)
    print("README.md index rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
