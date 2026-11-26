from __future__ import annotations

"""Download a reproducible, randomly mixed arXiv sample into data/inbox/<topic>.

The topic quotas are drawn from a Dirichlet distribution (seeded for reproducibility).
Within each arXiv primary category, papers are randomly sampled from a recent candidate
pool. The resulting folder name is retained as a weak source-topic label for evaluation.
"""

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ATOM = "http://www.w3.org/2005/Atom"
ARXIV = "http://arxiv.org/schemas/atom"
OPENSEARCH = "http://a9.com/-/spec/opensearch/1.1/"

@dataclass(frozen=True)
class Topic:
    category: str
    slug: str
    label: str

TOPICS = [
    Topic("cs.AI", "artificial_intelligence", "Artificial Intelligence"),
    Topic("cs.CL", "language_technology", "Language Technology"),
    Topic("cs.CV", "computer_vision", "Computer Vision"),
    Topic("cs.IR", "information_retrieval", "Information Retrieval"),
    Topic("cs.LG", "machine_learning", "Machine Learning"),
    Topic("cs.RO", "robotics", "Robotics"),
    Topic("eess.SP", "signal_processing", "Signal Processing"),
    Topic("math.OC", "optimization_control", "Optimization and Control"),
    Topic("q-bio.QM", "quantitative_biology", "Quantitative Biology"),
    Topic("stat.ML", "statistical_learning", "Statistical Learning"),
    Topic("econ.GN", "economics_general", "Economics"),
]


def random_quotas(total: int, topics: list[Topic], seed: int, concentration: float = 20.0) -> dict[str, int]:
    """Seeded non-uniform topic allocation; every topic receives at least 1 paper."""
    if total < len(topics):
        raise ValueError(f"target must be >= number of topics ({len(topics)})")
    rng = random.Random(seed)
    # Gamma sampling avoids requiring numpy just to run the downloader.
    weights = [rng.gammavariate(concentration, 1.0) for _ in topics]
    s = sum(weights)
    exact = [total * w / s for w in weights]
    quotas = [max(1, int(x)) for x in exact]
    while sum(quotas) > total:
        candidates = [i for i, q in enumerate(quotas) if q > 1]
        i = min(candidates, key=lambda j: (exact[j] - quotas[j], -quotas[j]))
        quotas[i] -= 1
    remainders = sorted(range(len(topics)), key=lambda i: exact[i] - int(exact[i]), reverse=True)
    for i in remainders[: total - sum(quotas)]:
        quotas[i] += 1
    return {topic.slug: quota for topic, quota in zip(topics, quotas, strict=True)}


def _text(entry: ET.Element, tag: str) -> str:
    node = entry.find(f"{{{ATOM}}}{tag}")
    return (node.text or "").strip() if node is not None else ""


def parse_feed(xml_bytes: bytes) -> tuple[list[dict[str, Any]], int]:
    root = ET.fromstring(xml_bytes)
    total_node = root.find(f"{{{OPENSEARCH}}}totalResults")
    total = int(total_node.text) if total_node is not None and total_node.text else 0
    rows: list[dict[str, Any]] = []
    for entry in root.findall(f"{{{ATOM}}}entry"):
        raw_id = _text(entry, "id").rstrip("/").split("/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", raw_id)
        primary = entry.find(f"{{{ARXIV}}}primary_category")
        primary_category = primary.attrib.get("term", "") if primary is not None else ""
        categories = [x.attrib.get("term", "") for x in entry.findall(f"{{{ATOM}}}category")]
        pdf_url = "https://arxiv.org/pdf/" + arxiv_id
        for link in entry.findall(f"{{{ATOM}}}link"):
            if link.attrib.get("title", "").lower() == "pdf":
                pdf_url = link.attrib.get("href", pdf_url)
                break
        authors = [
            (a.find(f"{{{ATOM}}}name").text or "").strip()
            for a in entry.findall(f"{{{ATOM}}}author")
            if a.find(f"{{{ATOM}}}name") is not None
        ]
        doi_node = entry.find(f"{{{ARXIV}}}doi")
        doi_value = (doi_node.text or "").strip() if doi_node is not None else ""
        rows.append({
            "arxiv_id": arxiv_id,
            "title": re.sub(r"\s+", " ", _text(entry, "title")),
            "abstract": re.sub(r"\s+", " ", _text(entry, "summary")),
            "published": _text(entry, "published"),
            "updated": _text(entry, "updated"),
            "primary_category": primary_category,
            "categories": categories,
            "authors": authors,
            "pdf_url": pdf_url,
            "doi": doi_value or None,
        })
    return rows, total


def request_bytes(url: str, user_agent: str, timeout: int = 60, retries: int = 5) -> bytes:
    headers = {"User-Agent": user_agent, "Accept": "application/atom+xml,application/pdf,*/*"}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            wait = min(60, 2 ** attempt * 2) + random.random()
            print(f"  request failed ({exc}); retry in {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Could not retrieve {url}: {last}")


def fetch_candidates(topic: Topic, needed: int, user_agent: str, delay: float,
                     max_candidates: int, seed: int, page_size: int = 500) -> list[dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    start = 0
    total_available = None
    while len(collected) < min(max_candidates, max(needed * 2, needed + 100)):
        count = min(page_size, max_candidates - start)
        if count <= 0:
            break
        params = urllib.parse.urlencode({
            "search_query": f"cat:{topic.category}", "start": start,
            "max_results": count, "sortBy": "submittedDate", "sortOrder": "descending",
        })
        url = "https://export.arxiv.org/api/query?" + params
        rows, total = parse_feed(request_bytes(url, user_agent))
        total_available = total
        for row in rows:
            # Keep only primary-category matches so the source-folder label is
            # an explicit, reproducible weak label rather than a cross-list guess.
            if row["primary_category"] == topic.category:
                collected[row["arxiv_id"]] = row
        start += len(rows)
        if not rows or start >= min(total, max_candidates):
            break
        time.sleep(delay)
    values = list(collected.values())
    random.Random(seed).shuffle(values)
    time.sleep(delay)
    return values


def read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []


def write_manifest(rows: list[dict[str, Any]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if rows:
        fields = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                                 for k, v in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a reproducible random-topic arXiv sample.")
    parser.add_argument("--target", type=int, default=1870)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--concentration", type=float, default=20.0,
                        help="Higher values make topic counts more balanced; lower values increase imbalance.")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between requests to arXiv. Keep >=3 unless you have a good reason.")
    parser.add_argument("--max-candidates-per-topic", type=int, default=2000)
    parser.add_argument("--email", default="", help="Optional contact email included in User-Agent.")
    args = parser.parse_args()
    if args.target < len(TOPICS) or args.delay < 0 or args.concentration <= 0:
        parser.error("invalid target, delay, or concentration")

    inbox = ROOT / "data" / "inbox"
    metadata_dir = ROOT / "data" / "metadata"
    plan_path = metadata_dir / "random_topic_distribution.json"
    manifest_jsonl = metadata_dir / "arxiv_random_sample.jsonl"
    manifest_csv = metadata_dir / "arxiv_random_sample.csv"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    user_agent = "ScientificIntelligenceLab-ResearchSampler/1.0" + (f" (mailto:{args.email})" if args.email else "")

    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("target") != args.target or plan.get("seed") != args.seed:
            raise SystemExit("Saved sample plan has a different target/seed. Archive or remove data/inbox topic folders and data/metadata before starting a different sample.")
        quotas = plan["quotas"]
    else:
        quotas = random_quotas(args.target, TOPICS, args.seed, args.concentration)
        plan = {"target": args.target, "seed": args.seed, "concentration": args.concentration,
                "sampling": "Dirichlet-random topic quotas; random sample within recent arXiv primary-category candidates",
                "topics": [asdict(t) for t in TOPICS], "quotas": quotas}
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    existing = read_existing_manifest(manifest_jsonl)
    selected_ids = {r["arxiv_id"] for r in existing}
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in existing:
        by_slug.setdefault(row["topic_slug"], []).append(row)
    complete: list[dict[str, Any]] = list(existing)

    print("Randomized topic quotas:")
    for topic in TOPICS:
        print(f"  {topic.label:28s} {quotas[topic.slug]:4d}")
    print(f"\nResuming existing manifest: {len(existing)} papers")

    for topic in TOPICS:
        have = len(by_slug.get(topic.slug, []))
        need = max(0, quotas[topic.slug] - have)
        if need == 0:
            continue
        print(f"\n[{topic.category}] need {need} more papers")
        candidates = fetch_candidates(topic, need, user_agent, args.delay,
                                      max_candidates=args.max_candidates_per_topic,
                                      seed=args.seed + sum(ord(ch) for ch in topic.category))
        if len(candidates) < need:
            print(f"  WARNING: only {len(candidates)} primary-category candidates found")
        downloaded = 0
        for candidate in candidates:
            if downloaded >= need:
                break
            arxiv_id = candidate["arxiv_id"]
            if arxiv_id in selected_ids:
                continue
            folder = inbox / topic.slug
            folder.mkdir(parents=True, exist_ok=True)
            pdf_path = folder / f"{arxiv_id.replace('/', '_')}.pdf"
            try:
                if not pdf_path.exists():
                    pdf_bytes = request_bytes(candidate["pdf_url"], user_agent, timeout=120)
                    if not pdf_bytes.startswith(b"%PDF"):
                        raise ValueError("response is not a PDF")
                    tmp = pdf_path.with_suffix(".pdf.part")
                    tmp.write_bytes(pdf_bytes)
                    tmp.replace(pdf_path)
                    time.sleep(args.delay)
                record = {
                    **candidate,
                    "topic_category": topic.category,
                    "topic_slug": topic.slug,
                    "topic_label": topic.label,
                    "quota": quotas[topic.slug],
                    "seed": args.seed,
                    "pdf_path": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
                }
                (folder / f"{arxiv_id.replace('/', '_')}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                complete.append(record)
                selected_ids.add(arxiv_id)
                downloaded += 1
                print(f"  {downloaded:4d}/{need}  {arxiv_id}  {candidate['title'][:90]}")
            except Exception as exc:
                print(f"  SKIP {arxiv_id}: {exc}", file=sys.stderr)
        by_slug[topic.slug] = [r for r in complete if r.get("topic_slug") == topic.slug]
        write_manifest(complete, manifest_jsonl, manifest_csv)
        print(f"  topic completed: {have + downloaded}/{quotas[topic.slug]}")

    # Stable order by topic then arXiv ID; this does not encode topic blocks in the PDF names.
    complete.sort(key=lambda r: (r.get("topic_slug", ""), r.get("arxiv_id", "")))
    write_manifest(complete, manifest_jsonl, manifest_csv)
    actual = len({r["arxiv_id"] for r in complete})
    summary = {"target": args.target, "downloaded": actual, "complete": actual >= args.target,
               "seed": args.seed, "quotas": quotas, "manifest_csv": str(manifest_csv)}
    (metadata_dir / "download_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if actual >= args.target else 2

if __name__ == "__main__":
    raise SystemExit(main())
