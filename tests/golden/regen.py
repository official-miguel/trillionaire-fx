"""Regenerate golden expected files (an explicit, reviewed step - never automatic).

    .venv\\Scripts\\python.exe -m tests.golden.regen

Bump ``tfx.versioning.RULE_VERSIONS`` for any strategy whose output changed, then
re-run validation (``.\\make.ps1 validate``) because the rule hash will have changed.
"""

from __future__ import annotations

from tests.golden.cases import EXPECTED_DIR, cases, compute, dump, header


def main() -> None:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in cases().items():
        (EXPECTED_DIR / f"{name}.json").write_text(
            dump({**header(), "outputs": compute(df)}) + "\n", encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
