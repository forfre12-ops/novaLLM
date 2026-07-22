"""공개 배포 전 OC(open.law.go.kr) API 키 레닥션 — provenance는 유지, 키만 가린다.

현재 코퍼스/SFT의 source_url에는 `OC=<개인키>`가 평문으로 박혀 있다(예: `?OC=tio&target=…`).
공개(HF/GitHub) 배포 전 이 값을 `OC=REDACTED`로 치환해야 키가 유출되지 않는다. law_id·시행일 등
다른 provenance는 그대로 둔다.

    python scripts/data/redact_export.py --in data/processed/laws.json --out data/public/laws.public.json
    python scripts/data/redact_export.py --check data/public/laws.public.json   # 잔존 키 0 검증(CI 게이트)
    python scripts/data/redact_export.py --selftest                              # 네트워크·데이터 불요

주의: 이 스크립트는 도구다. 실제 공개(레닥션본을 public 경로에 커밋/업로드)는 별도 결정이다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def safe_print(text: str = "") -> None:
    """Windows cp949 콘솔에서도 출력 불가 문자로 죽지 않게 치환 출력."""
    import sys as _sys
    enc = _sys.stdout.encoding or "utf-8"
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


# OC=<값>에서 값(구분자 전까지)을 REDACTED로. 이미 REDACTED면 건드리지 않는다.
_OC_RE = re.compile(r"OC=(?!REDACTED\b)[^&\"'\s\\]+")
# 잔존 키 탐지용: REDACTED가 아닌 OC=값
_OC_LEAK_RE = re.compile(r"OC=(?!REDACTED\b)[^&\"'\s\\]+")


def redact(text: str) -> str:
    return _OC_RE.sub("OC=REDACTED", text)


def find_leaks(text: str) -> list[str]:
    return _OC_LEAK_RE.findall(text)


def _selftest() -> int:
    sample = '{"source_url":"https://www.law.go.kr/DRF/lawService.do?OC=tio&target=eflaw&MST=61603"}'
    red = redact(sample)
    leaks = find_leaks(red)
    ok = "OC=REDACTED" in red and "OC=tio" not in red and not leaks
    # 이미 레닥션된 것은 재치환/오탐 없음
    idem = redact(red) == red and not find_leaks(red)
    safe_print(f"  redact: {'PASS' if ok else 'FAIL'} ({red[:60]}…)")
    safe_print(f"  idempotent/no-leak: {'PASS' if idem else 'FAIL'}")
    safe_print(f"\nredact_export selftest: {'PASS' if ok and idem else 'FAIL'}")
    return 0 if ok and idem else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input")
    ap.add_argument("--out")
    ap.add_argument("--check", help="파일에 OC 키 잔존이 없는지 검증(있으면 exit 1)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()

    if args.check:
        text = Path(args.check).read_text(encoding="utf-8")
        leaks = find_leaks(text)
        if leaks:
            safe_print(f"redact check: FAIL — OC 키 잔존 {len(leaks)}건 (예: {sorted(set(leaks))[:3]})")
            return 1
        safe_print(f"redact check: PASS — {args.check}에 잔존 OC 키 없음")
        return 0

    if not args.input or not args.out:
        ap.error("--in 과 --out 이 필요합니다(또는 --check / --selftest).")
    text = Path(args.input).read_text(encoding="utf-8")
    n_before = len(find_leaks(text))
    red = redact(text)
    leaks_after = find_leaks(red)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(red, encoding="utf-8")
    safe_print(f"레닥션: {args.input} → {args.out} (OC 키 {n_before}건 → REDACTED)")
    if leaks_after:
        safe_print(f"  [경고] 잔존 {len(leaks_after)}건 — 패턴 확인 필요")
        return 1
    safe_print("  잔존 OC 키 0 확인")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
