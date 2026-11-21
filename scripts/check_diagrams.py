"""Verify committed architecture previews against freshly rendered PlantUML diagrams."""

from pathlib import Path
import re
import sys
from xml.etree import ElementTree


def embedded_source(path: Path) -> bytes:
    """Read the source metadata from a well-formed PlantUML SVG."""
    ElementTree.parse(path)
    match = re.search(rb"<\?plantuml-src\s+(\S+?)\s*\?>", path.read_bytes())

    if match is None:
        raise ValueError(f"{path}: missing PlantUML source metadata")

    return match[1]


def check_previews(rendered: Path, committed: Path) -> None:
    """Require matching names and source metadata while allowing platform-specific layout."""
    expected = {path.name for path in rendered.glob("*.svg")}
    actual = {path.name for path in committed.glob("*.svg")}

    if not expected or actual != expected:
        raise ValueError(f"SVG inventory mismatch: missing={expected - actual}, extra={actual - expected}")

    for name in sorted(expected):
        if embedded_source(committed / name) != embedded_source(rendered / name):
            raise ValueError(f"Stale diagram: regenerate {committed / name} from its .puml source")


if __name__ == "__main__":
    check_previews(Path(sys.argv[1]), Path("docs/architecture/diagrams/rendered"))
