"""Build the curated law eval seed from hand-selected IDs/questions.

The output files are tracked eval artifacts. Gold spans are extracted from the
normalized corpus so typos cannot silently enter the partial-span set.

    python scripts/data/build_curated_law_eval.py --corpus data/processed/laws.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSWERABLE = [
    ("대한민국헌법 제1조 ①", "대한민국의 국가 형태는 무엇인가?"),
    ("대한민국헌법 제1조 ②", "대한민국의 주권과 권력의 근원은 누구에게 있는가?"),
    ("대한민국헌법 제10조", "국민의 존엄, 가치, 행복추구권과 국가의 기본권 보장 의무는 어떻게 규정되는가?"),
    ("대한민국헌법 제11조 ①", "법 앞의 평등과 차별금지는 어떻게 규정되는가?"),
    ("대한민국헌법 제12조 ①", "신체의 자유와 적법절차 원칙은 어떻게 규정되는가?"),
    ("대한민국헌법 제21조 ①", "언론ㆍ출판 및 집회ㆍ결사의 자유는 어떻게 규정되는가?"),
    ("대한민국헌법 제23조 ①", "재산권 보장과 그 내용ㆍ한계는 어떻게 규정되는가?"),
    ("대한민국헌법 제27조 ④", "형사피고인의 무죄추정은 어떻게 규정되는가?"),
    ("대한민국헌법 제32조 ①", "근로의 권리와 최저임금제 시행 의무는 어떻게 규정되는가?"),
    ("대한민국헌법 제37조 ②", "국민의 자유와 권리를 제한할 수 있는 요건과 한계는 무엇인가?"),
    ("민법 제103조", "반사회질서의 법률행위는 어떻게 규정되는가?"),
    ("민법 제104조", "불공정한 법률행위는 어떻게 규정되는가?"),
    ("민법 제107조 ①", "진의 아닌 의사표시는 원칙적으로 어떤 효력이 있는가?"),
    ("민법 제108조 ①", "상대방과 통정한 허위 의사표시는 어떤 효력이 있는가?"),
    ("민법 제741조", "부당이득 반환의 기본 요건은 어떻게 규정되는가?"),
    ("민법 제750조", "불법행위로 인한 손해배상책임은 어떻게 규정되는가?"),
    ("민법 제751조 ①", "정신상 고통에 대한 손해배상책임은 어떻게 규정되는가?"),
    ("민법 제756조 ①", "사용자의 배상책임은 어떻게 규정되는가?"),
    ("형법 제13조", "고의와 죄의 성립요소 인식은 어떻게 규정되는가?"),
    ("형법 제21조 ①", "정당방위는 어떤 경우에 벌하지 않는가?"),
    ("형법 제250조 ①", "사람을 살해한 경우의 형은 어떻게 규정되는가?"),
    ("형법 제257조 ①", "상해죄의 기본 구성요건과 형은 어떻게 규정되는가?"),
    ("형법 제347조 ①", "사기죄의 기본 구성요건과 형은 어떻게 규정되는가?"),
    ("형법 제355조 ①", "횡령죄의 기본 구성요건과 형은 어떻게 규정되는가?"),
    ("개인정보 보호법 제3조 ①", "개인정보처리자의 개인정보 처리 목적 명확화 의무는 어떻게 규정되는가?"),
    ("개인정보 보호법 제15조 ①", "개인정보처리자가 개인정보를 수집할 수 있는 경우는 어떻게 규정되는가?"),
    ("개인정보 보호법 제17조 ①", "개인정보처리자가 개인정보를 제3자에게 제공할 수 있는 경우는 무엇인가?"),
    ("개인정보 보호법 제21조 ①", "개인정보 보유기간 경과 후 파기 의무는 어떻게 규정되는가?"),
    ("전자금융거래법 제6조 ①", "접근매체 선정ㆍ사용ㆍ관리 시 준수사항은 어떻게 규정되는가?"),
    ("전자금융거래법 제9조 ①", "전자금융거래 오류로 인한 손해배상책임은 어떻게 규정되는가?"),
]

PARTIAL = [
    ("대한민국헌법 제10조", "국민의 행복추구권 부분만 인용하라.", 0),
    ("대한민국헌법 제10조", "국가의 기본권 보장 의무 부분만 인용하라.", 1),
    ("대한민국헌법 제11조 ①", "법 앞의 평등 원칙 부분만 인용하라.", 0),
    ("대한민국헌법 제11조 ①", "성별ㆍ종교ㆍ사회적 신분에 따른 차별금지 부분만 인용하라.", 1),
    ("대한민국헌법 제12조 ①", "신체의 자유를 가진다는 부분만 인용하라.", 0),
    ("대한민국헌법 제21조 ④", "언론ㆍ출판의 명예ㆍ권리 침해 금지 부분만 인용하라.", 0),
    ("대한민국헌법 제23조 ①", "재산권 보장 원칙 부분만 인용하라.", 0),
    ("대한민국헌법 제23조 ①", "재산권의 내용과 한계를 법률로 정한다는 부분만 인용하라.", 1),
    ("대한민국헌법 제27조 ③", "신속한 재판을 받을 권리 부분만 인용하라.", 0),
    ("대한민국헌법 제37조 ②", "자유와 권리 제한의 요건 부분만 인용하라.", 0),
    ("민법 제107조 ①", "진의 아닌 의사표시의 효력 부분만 인용하라.", 0),
    ("민법 제108조 ①", "통정한 허위 의사표시의 효력 부분만 인용하라.", 0),
    ("민법 제141조", "취소한 법률행위의 소급 무효 부분만 인용하라.", 0),
    ("민법 제390조", "채무불이행 손해배상책임 부분만 인용하라.", 0),
    ("민법 제393조 ①", "통상손해를 한도로 한다는 부분만 인용하라.", 0),
    ("민법 제750조", "불법행위 손해배상책임 부분만 인용하라.", 0),
    ("민법 제751조 ①", "정신상 고통에 대한 배상책임 부분만 인용하라.", 0),
    ("민법 제756조 ①", "피용자의 사무집행 관련 손해에 대한 사용자 책임 부분만 인용하라.", 0),
    ("형법 제13조", "죄의 성립요소 인식이 없는 행위 부분만 인용하라.", 0),
    ("형법 제21조 ①", "자기 또는 타인의 법익에 대한 현재의 부당한 침해 방위 부분만 인용하라.", 0),
    ("형법 제250조 ①", "사람을 살해한 경우의 형 부분만 인용하라.", 0),
    ("형법 제257조 ①", "사람의 신체를 상해한 경우의 형 부분만 인용하라.", 0),
    ("형법 제347조 ①", "기망으로 재산상 이익을 취득한 경우의 형 부분만 인용하라.", 0),
    ("개인정보 보호법 제3조 ①", "개인정보 처리 목적을 명확히 해야 한다는 부분만 인용하라.", 0),
    ("개인정보 보호법 제15조 ①", "정보주체의 동의를 받은 경우 수집 가능하다는 부분만 인용하라.", 0),
    ("개인정보 보호법 제17조 ①", "정보주체의 동의를 받은 경우 제3자 제공이 가능하다는 부분만 인용하라.", 0),
    ("개인정보 보호법 제21조 ①", "보유기간 경과 후 개인정보 파기 의무 부분만 인용하라.", 0),
    ("전자금융거래법 제6조 ①", "접근매체를 선정ㆍ사용ㆍ관리할 때 다른 법률 준수 의무 부분만 인용하라.", 0),
    ("전자금융거래법 제9조 ①", "전자금융거래 오류 손해에 대한 금융회사 책임 부분만 인용하라.", 0),
    ("전자금융거래법 제49조 ①", "접근매체 위조ㆍ변조 관련 벌칙 부분만 인용하라.", 0),
]

UNANSWERABLE = [
    "특허법상 특허출원 심사청구 기간은 어떻게 규정되어 있는가?",
    "주택임대차보호법상 대항력 발생 요건은 어떻게 규정되어 있는가?",
    "상가건물 임대차보호법상 계약갱신 요구권 행사기간은 어떻게 규정되어 있는가?",
    "저작권법상 공정이용 판단 요소는 무엇인가?",
    "근로기준법상 연차 유급휴가 산정 방식은 어떻게 규정되어 있는가?",
    "민사소송법상 상고 제기 절차는 어떻게 규정되어 있는가?",
    "국가공무원법상 공무원의 결격사유는 어떻게 규정되어 있는가?",
    "형사소송법상 구속영장 청구 절차는 어떻게 규정되어 있는가?",
    "도로교통법상 운전면허 취소 사유는 어떻게 규정되어 있는가?",
    "상법상 상행위와 상인 자격은 어떻게 규정되어 있는가?",
    "행정절차법상 처분 사전통지의 예외는 무엇인가?",
    "국세기본법상 경정청구 기간은 어떻게 규정되어 있는가?",
    "부동산등기법상 등기신청 각하 사유는 어떻게 규정되어 있는가?",
    "노동조합법상 부당노동행위 구제절차는 어떻게 규정되어 있는가?",
    "정보통신망법상 임시조치 절차는 어떻게 규정되어 있는가?",
    "청소년보호법상 청소년유해매체물 표시의무는 어떻게 규정되어 있는가?",
    "의료법상 진료기록부 보존기간은 어떻게 규정되어 있는가?",
    "건축법상 건축허가 대상 건축물은 어떻게 규정되어 있는가?",
    "공정거래법상 시장지배적 사업자의 남용행위는 어떻게 규정되어 있는가?",
    "전자문서법상 공인전자문서센터의 지정 요건은 어떻게 규정되어 있는가?",
]


def load_articles(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("articles", data)


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=\.)\s+", text)
    return [p for p in parts if p]


def assert_ids_exist(articles: dict[str, str]) -> None:
    missing = [cid for cid, _ in ANSWERABLE if cid not in articles]
    missing += [cid for cid, _, _ in PARTIAL if cid not in articles]
    if missing:
        raise SystemExit("missing curated IDs: " + ", ".join(sorted(set(missing))))


def build_partial_items(articles: dict[str, str]) -> list[dict]:
    out = []
    for cid, question, sent_idx in PARTIAL:
        spans = split_sentences(articles[cid])
        if sent_idx >= len(spans):
            raise SystemExit(f"sentence index out of range: {cid} idx={sent_idx} n={len(spans)}")
        span = spans[sent_idx]
        if span not in articles[cid]:
            raise SystemExit(f"span is not exact substring: {cid}")
        out.append({"id": cid, "question": question, "gold_span": span})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/processed/laws.json")
    ap.add_argument("--questions-out", default="eval/questions.laws.curated.json")
    ap.add_argument("--partial-out", default="eval/questions.partial.laws.curated.json")
    ap.add_argument("--unanswerable-out", default="eval/questions.unanswerable.laws.curated.json")
    args = ap.parse_args()

    articles = load_articles(Path(args.corpus))
    assert_ids_exist(articles)

    questions = {cid: question for cid, question in ANSWERABLE}
    partial = build_partial_items(articles)

    Path(args.questions_out).write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.partial_out).write_text(
        json.dumps(partial, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.unanswerable_out).write_text(
        json.dumps(UNANSWERABLE, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved answerable: {args.questions_out} ({len(questions)})")
    print(f"saved partial: {args.partial_out} ({len(partial)})")
    print(f"saved unanswerable: {args.unanswerable_out} ({len(UNANSWERABLE)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
