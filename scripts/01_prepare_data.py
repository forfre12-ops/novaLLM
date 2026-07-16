"""01_prepare_data.py — 원시 데이터 → 학습용 chat jsonl 변환

data/raw/ 의 원시 데이터를 {"messages": [...]} 형식 jsonl로 정규화해
data/processed/train.jsonl 로 저장한다.

한국어 SFT 데이터 출처 예:
  - 공개 instruction 셋 (KoAlpaca, KULLM 등 — 라이선스 확인 필수)
  - 강한 모델로 합성한 한국어 Q&A (제공자 이용약관 확인)
  - 도메인 문서 기반 자체 Q&A (차별화 해자)

실행:
    python scripts/01_prepare_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/train.jsonl")


def to_chat(instruction: str, output: str, system: str | None = None) -> dict:
    """instruction/output 쌍을 chat messages 포맷으로 변환."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})
    messages.append({"role": "assistant", "content": output})
    return {"messages": messages}


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        # TODO: data/raw 의 실제 파일 포맷에 맞게 파싱 로직을 작성하세요.
        # 아래는 alpaca 스타일(json 배열) 예시입니다.
        for path in sorted(RAW_DIR.glob("*.json")):
            rows = json.loads(path.read_text(encoding="utf-8"))
            for r in rows:
                instr = r.get("instruction", "")
                inp = r.get("input", "")
                if inp:
                    instr = f"{instr}\n\n{inp}"
                rec = to_chat(instr, r.get("output", ""))
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    if n == 0:
        print("경고: 변환된 데이터 0건. data/raw 에 원시 데이터를 넣고 파서를 작성하세요.")
    else:
        print(f"{n}건 변환 완료 → {OUT_FILE}")


if __name__ == "__main__":
    main()
