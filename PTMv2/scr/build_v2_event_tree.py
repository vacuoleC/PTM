"""E7: parse PTMv2 whatwedo.md event tree into structured CSV + SVG.

Extracts event IDs, parent relationships, and status from the ledger,
then emits v2_task_comparison.csv and v2_event_tree.svg (standard library
only, no external deps).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_events(ledger_text: str) -> list[dict]:
    """Parse '### <ID> — <title>' headers plus status/parent lines."""
    events = []
    current = None
    for line in ledger_text.splitlines():
        m = re.match(r"^###(?!##)\s+([A-Z0-9.]+(?:\.[a-z])?)\s*[—\-]\s*(.+)$", line)
        if m:
            current = {"event_id": m.group(1), "title": m.group(2).strip(), "status": "unknown",
                       "parent": None, "why": "", "result": ""}
            events.append(current)
            continue
        if current is None:
            continue
        s = re.search(r"状态[：:]\s*(\S+)", line)
        if s:
            current["status"] = s.group(1)
        p = re.search(r"父事件[：:]\s*([A-Z0-9.]+)", line)
        if p:
            current["parent"] = p.group(1)

    # Parent chain: assign to first event with matching ID prefix or explicit parent
    by_id = {e["event_id"]: e for e in events}
    for e in events:
        if e["parent"] and e["parent"] in by_id:
            continue
        # Infer parent: drop trailing .x or -suffix
        base = re.sub(r"(\.[a-z])?$", "", e["event_id"])
        base = re.sub(r"-[a-z0-9]+$", "", base)
        if base in by_id and base != e["event_id"]:
            e["parent"] = base
    return events


def build_tree(events: list[dict]) -> dict:
    """Return adjacency {event_id: [children]}, cycle-safe."""
    by_id = {e["event_id"]: e for e in events}
    tree: dict[str, list[str]] = {}
    for e in events:
        tree.setdefault(e["event_id"], [])
    for e in events:
        p = e["parent"]
        # Skip self-references and unknown parents (cycle-safe)
        if p and p in tree and p != e["event_id"]:
            tree[p].append(e["event_id"])
    return tree


def to_svg(tree: dict, events: list[dict], out: Path) -> None:
    """Layered SVG tree (standard library XML)."""
    by_id = {e["event_id"]: e for e in events}
    depth: dict[str, int] = {}
    order: list[str] = []

    def walk(nid, d):
        depth[nid] = d
        order.append(nid)
        for c in sorted(tree.get(nid, [])):
            walk(c, d + 1)

    roots = [e["event_id"] for e in events if e["parent"] is None or e["parent"] not in by_id]
    for r in sorted(roots):
        walk(r, 0)

    max_depth = max(depth.values(), default=0)
    n = len(order)
    W, H = 1400, max(300, n * 34 + 40)
    x_of = {nid: 60 + depth[nid] * 220 for nid in order}
    y_of = {nid: 30 + i * 34 for i, nid in enumerate(order)}

    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", width=str(W), height=str(H),
                     viewBox=f"0 0 {W} {H}")
    # Edges
    for nid in order:
        for c in tree.get(nid, []):
            x1, y1 = x_of[nid] + 150, y_of[nid] + 9
            x2, y2 = x_of[c], y_of[c] + 9
            ET.SubElement(svg, "line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),
                          stroke="#888", stroke_width="1")
    # Nodes
    colors = {"已完成": "#4caf50", "进行中": "#ff9800", "阻塞": "#f44336",
              "unknown": "#9e9e9e"}
    for nid in order:
        e = by_id[nid]
        color = colors.get(e["status"], "#9e9e9e")
        x, y = x_of[nid], y_of[nid]
        ET.SubElement(svg, "rect", x=str(x), y=str(y), width="150", height="18",
                      rx="4", fill=color, opacity="0.85")
        label = f"{nid}: {e['title'][:28]}"
        ET.SubElement(svg, "text", x=str(x + 4), y=str(y + 13), font_size="11",
                      fill="#111").text = label
    tree_el = ET.ElementTree(svg)
    tree_el.write(out, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ledger = (ROOT / "whatwedo.md").read_text(encoding="utf-8")
    events = parse_events(ledger)
    tree = build_tree(events)

    # CSV
    rows = [{"event_id": e["event_id"], "title": e["title"], "status": e["status"],
             "parent": "" if e["parent"] == e["event_id"] else (e["parent"] or ""),
             "why": e["why"][:80], "result": e["result"][:80]}
            for e in events]
    df = pd.DataFrame(rows)
    csv_out = ROOT / "outputs/tables/v2_task_comparison.csv"
    df.to_csv(csv_out, index=False)

    # SVG
    svg_out = ROOT / "outputs/figures/v2_event_tree.svg"
    to_svg(tree, events, svg_out)

    print(f"parsed {len(events)} events, {len(tree)} nodes")
    print(f"wrote {csv_out}")
    print(f"wrote {svg_out}")


if __name__ == "__main__":
    main()
