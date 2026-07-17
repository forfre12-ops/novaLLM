"""실험 재현성 메타데이터 헬퍼 — 모든 result JSON에 공통 provenance를 기록한다.

'측정을 소유한다' 전략에서 외부 검증 가능성은 최소 요건이다. 결과 수치만 저장하면
어떤 모델·어댑터·코퍼스·시드·스코어러 버전으로 나왔는지 재구성할 수 없다. 이 헬퍼는
그 공백을 닫는다(engineering 감사 반영).

    from run_meta import run_metadata, sha256_file, sha256_text
    meta = run_metadata(models={...}, corpus_path=..., seed=..., extra={...})
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path


def sha256_file(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    if p.is_dir():
        # 디렉터리(어댑터 폴더)는 파일별 해시를 정렬 결합
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.relative_to(p).as_posix().encode())
                h.update(f.read_bytes())
        return h.hexdigest()
    h.update(p.read_bytes())
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _pkg_versions() -> dict[str, str]:
    vers: dict[str, str] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = "n/a"
    return vers


def run_metadata(
    *,
    models: dict[str, str],
    corpus_path: str | None = None,
    questions_path: str | None = None,
    adapter_path: str | None = None,
    seed: int | None = None,
    scorer_version: str | None = None,
    extra: dict | None = None,
) -> dict:
    """result JSON에 넣을 provenance 블록. 타임스탬프는 워크플로 재현성을 위해 호출측에서 주입."""
    meta = {
        "git_rev": _git_rev(),
        "packages": _pkg_versions(),
        "models": models,
        "seed": seed,
        "scorer_version": scorer_version,
        "adapter_path": adapter_path,
        "adapter_sha256": sha256_file(adapter_path) if adapter_path else None,
        "corpus_path": corpus_path,
        "corpus_sha256": sha256_file(corpus_path) if corpus_path else None,
        "questions_path": questions_path,
        "questions_sha256": sha256_file(questions_path) if questions_path else None,
    }
    if extra:
        meta.update(extra)
    return meta
