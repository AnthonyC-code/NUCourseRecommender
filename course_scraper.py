import re, time, sys, requests, pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://catalogs.northwestern.edu"
INDEX_URL = f"{BASE}/undergraduate/courses-az/"
CSV_OUT = "nu_courses_2024_25.csv"

def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = "NUCourseScraper/0.5 (you@u.northwestern.edu)"
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(total=0, allowed_methods=frozenset(["GET"]))
        ),
    )
    return s


session = make_session()


def fetch(url: str, tries: int = 3, timeout: tuple = (10, 120)) -> str | None:
    """GET text or None after `tries` failures."""
    for attempt in range(tries):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    tqdm.write(f"⚠️  skipping {url}")
    return None


def get_subject_links() -> list[str]:
    html = fetch(INDEX_URL, tries=5)
    if not html:
        sys.exit("Could not fetch index page.")
    soup = BeautifulSoup(html, "html.parser")
    links = {
        BASE + a["href"]
        for a in soup.select("a[href^='/undergraduate/courses-az/']")
        if a["href"].rstrip("/") != "/undergraduate/courses-az"
    }
    return sorted(links)


header_rx = re.compile(
    r"^([A-Z_]+)\s+([\dA-Z\-]+-?\d*)\s+(.+?)\s+\((?:\d+(?:\.\d+)?|Variable) Unit"
)
def parse_subject(url: str) -> list[dict]:
    html = fetch(url)
    if not html:
        return []

    soup  = BeautifulSoup(html, "html.parser")
    lines = soup.get_text("\n", strip=True).splitlines()

    rows, current = [], None
    for line in lines:
        m = header_rx.match(line)
        if m:
            if current:
                rows.append(current)
            current = {
                "subject":     m.group(1),
                "catalog_num": m.group(2),
                "title_units": line,
                "description": "",
            }
        elif current and line and not line.startswith("Prerequisite"):
            # accumulate description until next header
            current["description"] += (" " if current["description"] else "") + line

    if current:
        rows.append(current)
    return rows


def main():
    all_rows = []
    for link in tqdm(get_subject_links(), desc="Subjects"):
        all_rows.extend(parse_subject(link))
        time.sleep(0.8)

    pd.DataFrame(all_rows).to_csv(CSV_OUT, index=False)
    tqdm.write(f"saved {len(all_rows)} rows → {CSV_OUT}")


if __name__ == "__main__":
    main()
