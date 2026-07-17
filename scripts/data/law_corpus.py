"""국가법령정보 OpenAPI 응답을 faithbench closed-set 코퍼스로 정규화한다.

목표 스키마는 기존 scorer가 바로 읽을 수 있는 `articles` mapping을 유지하면서,
다법령 확장에 필요한 provenance/snapshot metadata를 `entries`에 보존하는 것이다.

입력은 lawService JSON 또는 XML을 모두 허용한다. 법제처 응답은 endpoint/target에 따라
루트 키와 반복 노드 모양이 달라질 수 있으므로, 파서는 조문 필드명을 기준으로 보수적으로
탐색한다.

    python scripts/data/law_corpus.py --demo
    python scripts/data/law_corpus.py --in data/raw/law_service.json --out data/processed/law_corpus.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "law-corpus-v0.1"

_WS_RE = re.compile(r"\s+")
_LEADING_MARK_RE = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]\s*")
_PARA_MARK = {
    1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
    11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯", 17: "⑰", 18: "⑱", 19: "⑲", 20: "⑳",
}


def safe_print(text: str = "") -> None:
    enc = sys.stdout.encoding or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def norm_text(value: Any) -> str:
    """법령 본문 비교용 공백 정규화."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        value = str(value)
    return _WS_RE.sub(" ", str(value)).strip()


def clean_body(value: Any) -> str:
    """항/조문 본문 앞의 원문 번호 표식을 source id로 옮기기 위해 제거한다."""
    text = norm_text(value)
    return _LEADING_MARK_RE.sub("", text).strip()


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def first_value(node: dict[str, Any], *keys: str) -> str:
    for key in keys:
        if key in node:
            text = norm_text(node[key])
            if text:
                return text
    return ""


def xml_to_obj(text: str) -> dict[str, Any]:
    """ElementTree XML을 JSON과 비슷한 dict/list 구조로 변환한다."""

    def convert(elem: ET.Element) -> Any:
        children = list(elem)
        own_text = norm_text(elem.text)
        if not children:
            return own_text
        out: dict[str, Any] = {}
        if own_text:
            out["_text"] = own_text
        for child in children:
            value = convert(child)
            if child.tag in out:
                if not isinstance(out[child.tag], list):
                    out[child.tag] = [out[child.tag]]
                out[child.tag].append(value)
            else:
                out[child.tag] = value
        return out

    root = ET.fromstring(text)
    return {root.tag: convert(root)}


def load_payload(path: str | Path) -> tuple[Any, str]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    stripped = raw.lstrip()
    if stripped.startswith("<"):
        return xml_to_obj(raw), raw
    return json.loads(raw), raw


def _walk_dicts(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_dicts(item)


def _law_meta(payload: Any) -> dict[str, str]:
    meta: dict[str, str] = {}
    for node in _walk_dicts(payload):
        law_name = first_value(node, "법령명_한글", "법령명한글", "법령명", "법령명_약칭")
        if law_name and "law_name" not in meta:
            meta["law_name"] = law_name
        law_id = first_value(node, "법령ID", "법령일련번호", "법령키", "MST", "mst")
        if law_id and "law_id" not in meta:
            meta["law_id"] = law_id
        effective_date = first_value(node, "시행일자", "시행일", "시행일자문자열")
        if effective_date and "effective_date" not in meta:
            meta["effective_date"] = effective_date
        promulgation_date = first_value(node, "공포일자", "공포일")
        if promulgation_date and "promulgation_date" not in meta:
            meta["promulgation_date"] = promulgation_date
    meta.setdefault("law_name", "법령")
    return meta


def _article_nodes(payload: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in _walk_dicts(payload):
        if "조문번호" in node or "조문내용" in node or "조문제목" in node:
            nodes.append(node)
    return nodes


def _article_label(number: str, branch: str = "") -> str:
    num = str(int(number)) if str(number).isdigit() else norm_text(number)
    br = norm_text(branch)
    if br and br not in {"0", "00", "000"}:
        br = str(int(br)) if br.isdigit() else br
        return f"제{num}조의{br}"
    return f"제{num}조"


def _para_label(value: Any, fallback_index: int) -> str:
    text = norm_text(value)
    if text:
        try:
            n = int(text)
            return _PARA_MARK.get(n, f"({n})")
        except ValueError:
            return text
    return _PARA_MARK.get(fallback_index, f"({fallback_index})")


def _join_child_units(node: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("호", "목"):
        for child in ensure_list(node.get(key)):
            if isinstance(child, dict):
                content = first_value(child, f"{key}내용", "내용", "_text")
                if content:
                    chunks.append(clean_body(content))
    return " ".join(c for c in chunks if c)


def _entry(
    *,
    law_name: str,
    law_id: str,
    article_id: str,
    text: str,
    article_number: str,
    paragraph_number: str = "",
    effective_date: str = "",
    promulgation_date: str = "",
    source_url: str = "",
) -> dict[str, str]:
    return {
        "id": article_id,
        "law_name": law_name,
        "law_id": law_id,
        "article_number": article_number,
        "paragraph_number": paragraph_number,
        "text": text,
        "effective_date": effective_date,
        "promulgation_date": promulgation_date,
        "source_url": source_url,
    }


def normalize_law_payload(payload: Any, *, source_url: str = "", raw_text: str = "") -> dict[str, Any]:
    """lawService 응답을 scorer-compatible closed-set corpus로 변환한다."""
    meta = _law_meta(payload)
    law_name = meta["law_name"]
    law_id = meta.get("law_id", "")
    effective_date = meta.get("effective_date", "")
    promulgation_date = meta.get("promulgation_date", "")
    entries: list[dict[str, str]] = []

    for node in _article_nodes(payload):
        article_no = first_value(node, "조문번호")
        if not article_no:
            continue
        if first_value(node, "조문여부") in {"전문", "부칙"}:
            continue

        branch = first_value(node, "조문가지번호")
        article_label = _article_label(article_no, branch)
        article_content = clean_body(first_value(node, "조문내용", "내용", "_text"))
        article_extra = _join_child_units(node)
        if article_extra:
            article_content = norm_text(f"{article_content} {article_extra}")

        paras = ensure_list(node.get("항"))
        if paras:
            for idx, para in enumerate(paras, 1):
                if not isinstance(para, dict):
                    continue
                para_no = _para_label(para.get("항번호"), idx)
                para_content = clean_body(first_value(para, "항내용", "내용", "_text"))
                para_extra = _join_child_units(para)
                text = norm_text(f"{para_content} {para_extra}") if para_extra else para_content
                if not text or text == "삭제":
                    continue
                cid = f"{law_name} {article_label} {para_no}"
                entries.append(_entry(
                    law_name=law_name,
                    law_id=law_id,
                    article_id=cid,
                    article_number=article_label,
                    paragraph_number=para_no,
                    text=text,
                    effective_date=effective_date,
                    promulgation_date=promulgation_date,
                    source_url=source_url,
                ))
        elif article_content and article_content != "삭제":
            cid = f"{law_name} {article_label}"
            entries.append(_entry(
                law_name=law_name,
                law_id=law_id,
                article_id=cid,
                article_number=article_label,
                text=article_content,
                effective_date=effective_date,
                promulgation_date=promulgation_date,
                source_url=source_url,
            ))

    articles = {e["id"]: e["text"] for e in entries}
    raw_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": f"{law_name} 국가법령정보 OpenAPI snapshot",
        "provenance": "open.law.go.kr lawService normalized by scripts/data/law_corpus.py",
        "authoritative_source": "국가법령정보 OpenAPI",
        "license": "법령 텍스트 = 저작권법 제7조 비보호",
        "closed_set": True,
        "snapshot_date": date.today().isoformat(),
        "source_url": source_url,
        "raw_sha256": raw_sha,
        "n_entries": len(entries),
        "articles": articles,
        "entries": entries,
    }


DEMO_PAYLOAD = {
    "법령": {
        "기본정보": {
            "법령명_한글": "예시법",
            "법령ID": "DEMO",
            "시행일자": "20260718",
            "공포일자": "20260701",
        },
        "조문": {
            "조문단위": [
                {
                    "조문번호": "1",
                    "조문내용": "제1조(목적) 이 법은 예시를 목적으로 한다.",
                },
                {
                    "조문번호": "2",
                    "조문내용": "제2조(인용)",
                    "항": [
                        {"항번호": "1", "항내용": "① 인용문은 원문 그대로 옮긴다."},
                        {"항번호": "2", "항내용": "② 근거가 없으면 답변하지 아니한다."},
                    ],
                },
            ]
        },
    }
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input")
    ap.add_argument("--out")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        corpus = normalize_law_payload(DEMO_PAYLOAD, source_url="https://example.test/law")
        ok = (
            corpus["n_entries"] == 3
            and corpus["articles"]["예시법 제2조 ①"] == "인용문은 원문 그대로 옮긴다."
            and corpus["articles"]["예시법 제2조 ②"] == "근거가 없으면 답변하지 아니한다."
        )
        safe_print(json.dumps(corpus, ensure_ascii=False, indent=2))
        safe_print(f"law_corpus demo: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if not args.input:
        ap.error("--in 또는 --demo 가 필요합니다.")
    payload, raw = load_payload(args.input)
    corpus = normalize_law_payload(payload, source_url=args.source_url, raw_text=raw)
    text = json.dumps(corpus, ensure_ascii=False, indent=2)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(text + "\n", encoding="utf-8")
        safe_print(f"saved: {outp} ({corpus['n_entries']} entries)")
    else:
        safe_print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
