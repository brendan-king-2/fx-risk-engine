"""Inject results.json into the dashboard template -> dashboard/index.html (self-contained)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    data = (HERE / "results.json").read_text()
    html = (HERE / "template.html").read_text().replace("/*__DATA__*/null", data, 1)
    (HERE / "index.html").write_text(html)
    print(f"wrote {HERE / 'index.html'} ({len(html)/1e3:.0f} KB)")
