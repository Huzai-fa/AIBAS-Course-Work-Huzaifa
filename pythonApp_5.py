#!/usr/bin/env python3
"""
scrape_github_readme.py

Usage:
    python scrape_github_readme.py

What it does:
 - Tries to fetch the raw README from raw.githubusercontent.com
 - If raw fetch fails, downloads the GitHub HTML page and extracts the rendered README area
 - Saves:
     - README.md        (raw markdown)
     - README.txt       (plain-text fallback/stripped)
     - readme_summary.json (list of top-level headings and code blocks)
Dependencies:
 - requests
 - beautifulsoup4
 - markdown (optional; used to convert md to text)
Install deps:
 pip install requests beautifulsoup4 markdown
"""

import json
import os
import re
import sys
from typing import List

import requests
from bs4 import BeautifulSoup

# === CONFIG ===
GITHUB_HTML_URL = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"
RAW_URL = "https://github.com/MarcusGrum/AIBAS/blob/main/README.md"
TIMEOUT = 15.0
OUT_DIR = "scraped_readme"
# =============

os.makedirs(OUT_DIR, exist_ok=True)


def fetch_raw_markdown(url: str) -> str:
    """Try to fetch raw markdown directly (raw.githubusercontent.com)."""
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    # Return text as-is (markdown)
    return resp.text


def fetch_github_html(url: str) -> str:
    """Fetch the GitHub page HTML (rendered)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ReadmeScraper/1.0; +https://example.org)"
    }
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.text


def extract_markdown_from_github_html(html: str) -> str:
    """
    Extract the README content from GitHub HTML. GitHub renders markdown inside
    <article class="markdown-body"> (or similar). We attempt to extract that
    section and convert it back into a reasonable markdown-like/plain text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Try several common selectors for the rendered markdown section
    candidates = []
    selectors = [
        "article.markdown-body",  # common for GitHub
        "div.markdown-body",      # alternate
        "div#readme",             # sometimes the #readme container
    ]
    for sel in selectors:
        found = soup.select(sel)
        if found:
            candidates.extend(found)

    if not candidates:
        # last resort: try to find main README by heading if present
        article = soup.find("article")
        if article:
            candidates = [article]

    if not candidates:
        raise RuntimeError("Could not find rendered README content in GitHub HTML.")

    # Take the first candidate
    readme_node = candidates[0]

    # We will reconstruct a plain-text / markdown approximation:
    # - Convert headings
    # - Keep code blocks
    # - Keep lists
    lines: List[str] = []

    def flush_text(text: str):
        # simple normalize: collapse multiple spaces, strip trailing spaces
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        for ln in text.splitlines():
            ln = ln.rstrip()
            if ln:
                lines.append(ln)

    for child in readme_node.descendants:
        # Handle code blocks (pre > code)
        if child.name == "pre":
            code = child.get_text()
            lines.append("```")
            lines.extend(code.rstrip("\n").splitlines())
            lines.append("```")
            # skip other handling for this child
            continue
        # handle headings
        if child.name and re.fullmatch(r"h[1-6]", child.name.lower()):
            level = int(child.name[1])
            heading_text = child.get_text(strip=True)
            lines.append("#" * level + " " + heading_text)
            continue
        # handle list items
        if child.name in ("li", "p", "span", "strong", "em", "a", "code"):
            # we'll let plain text be collected
            text = child.get_text(separator=" ", strip=True)
            if text:
                lines.append(text)

    # Post-process lines to remove duplicates and join sensibly
    # We try to preserve code fence blocks already added
    combined = []
    prev = None
    for ln in lines:
        if ln == prev:
            continue
        combined.append(ln)
        prev = ln

    return "\n\n".join(combined)


def extract_headings_and_codeblocks_from_markdown(md: str):
    """
    Simple markdown parsing for:
     - headings (lines starting with #)
     - fenced code blocks (triple backticks)
    """
    headings = []
    code_blocks = []

    # headings
    for ln in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            headings.append({"level": level, "text": text})

    # code blocks
    fence_pattern = re.compile(r"^```")
    in_code = False
    cur_code_lines = []
    for ln in md.splitlines():
        if fence_pattern.match(ln):
            if in_code:
                # close
                code_blocks.append("\n".join(cur_code_lines))
                cur_code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            cur_code_lines.append(ln)
    # safety: if file had non-closed block
    if in_code and cur_code_lines:
        code_blocks.append("\n".join(cur_code_lines))

    return headings, code_blocks


def save_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {path}")


def md_to_plain_text(md_text: str) -> str:
    """
    Convert markdown to plain text. If the `markdown` package is available,
    we use it to convert to HTML then strip tags. Otherwise, fallback to
    a minimalistic approach (strip #, backticks, etc).
    """
    try:
        import markdown as _md
        from bs4 import BeautifulSoup as _BS

        html = _md.markdown(md_text)
        soup = _BS(html, "html.parser")
        # remove script/style
        for s in soup(["script", "style"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        # collapse multiple blank lines
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return text
    except Exception:
        # minimal fallback
        txt = re.sub(r"```.*?```", "", md_text, flags=re.S)  # remove fenced code
        txt = re.sub(r"#+", "", txt)  # remove heading markers
        txt = txt.replace("`", "")
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()


def main():
    markdown_text = None

    # 1) Try direct raw URL
    try:
        print(f"Attempting raw fetch from: {RAW_URL}")
        markdown_text = fetch_raw_markdown(RAW_URL)
        print("Fetched raw markdown successfully.")
    except Exception as e_raw:
        print(f"Raw fetch failed: {e_raw!r}")
        # 2) Fall back to scraping GitHub HTML
        try:
            print(f"Falling back to scraping GitHub HTML page: {GITHUB_HTML_URL}")
            html = fetch_github_html(GITHUB_HTML_URL)
            markdown_text = extract_markdown_from_github_html(html)
            print("Extracted markdown-like content from GitHub HTML.")
        except Exception as e_html:
            print(f"Failed to extract README from GitHub HTML: {e_html!r}")
            print("Exiting with failure.")
            sys.exit(2)

    # Save raw markdown
    md_path = os.path.join(OUT_DIR, "README.md")
    save_file(md_path, markdown_text)

    # Save plain text
    text_path = os.path.join(OUT_DIR, "README.txt")
    plain = md_to_plain_text(markdown_text)
    save_file(text_path, plain)

    # Create and save a small summary JSON: headings + code blocks
    headings, code_blocks = extract_headings_and_codeblocks_from_markdown(markdown_text)
    summary = {
        "source_url": GITHUB_HTML_URL,
        "headings": headings,
        "num_code_blocks": len(code_blocks),
        "first_code_blocks_preview": code_blocks[:3],
    }
    json_path = os.path.join(OUT_DIR, "readme_summary.json")
    save_file(json_path, json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nDone. Outputs are in the directory:", OUT_DIR)


if __name__ == "__main__":
    main()
