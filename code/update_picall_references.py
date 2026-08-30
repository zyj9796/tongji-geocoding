"""根据《图片整理映射》同步代码和文档中的图片文件名引用。"""

from __future__ import annotations

import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
MANIFEST = WORKSPACE / "图片整理映射.md"
EDITABLE_SUFFIXES = {".py", ".sh", ".js", ".json", ".md", ".txt"}


def read_pairs() -> list[tuple[str, str]]:
    pairs = []
    pattern = re.compile(r"^- `([^`]+)` → `([^`]+)`$")
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs


def final_paths(pairs: list[tuple[str, str]]) -> dict[str, str]:
    direct = dict(pairs)
    resolved = {}
    for source in direct:
        target = direct[source]
        seen = {source}
        while target in direct and target not in seen:
            seen.add(target)
            target = direct[target]
        resolved[source] = target
    return resolved


def replacements(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    resolved = final_paths(pairs)
    values: dict[str, set[str]] = {}
    for source, target in resolved.items():
        values.setdefault(Path(source).name, set()).add(Path(target).name)
    output = list(resolved.items())
    output.extend(
        (source, next(iter(targets)))
        for source, targets in values.items()
        if len(targets) == 1 and source != next(iter(targets))
    )
    output.extend(
        [
            ("results/picall/注册复现", "results/picall/注册复现"),
            ("results/picall/主流程", "results/picall/主流程"),
            ("results/picall/正式图件", "results/picall/正式图件"),
            ("results/picall/过程图件", "results/picall/过程图件"),
            ("/results/picall/", "/results/picall/"),
        ]
    )
    return sorted(output, key=lambda item: len(item[0]), reverse=True)


def editable_files():
    excluded_names = {".git", "tmp", "data", "work", "node_modules", "__pycache__"}
    for path in WORKSPACE.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EDITABLE_SUFFIXES:
            continue
        relative_parts = path.relative_to(WORKSPACE).parts
        if any(part in excluded_names for part in relative_parts):
            continue
        if "results" in relative_parts and path.name != "README.md":
            continue
        if path == MANIFEST:
            continue
        yield path


def main() -> None:
    changes = replacements(read_pairs())
    changed_files = 0
    for path in editable_files():
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original
        for source, target in changes:
            updated = updated.replace(source, target)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed_files += 1
    print(f"changed_files={changed_files}")


if __name__ == "__main__":
    main()
