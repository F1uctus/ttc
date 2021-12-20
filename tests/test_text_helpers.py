from pathlib import Path


def find_test_texts(path: Path):
    return [e.name for e in path.iterdir() if e.suffix == ".txt"]


def load_test(path: Path, file_name: str, delimiter="-" * 20):
    """
    :returns: ("Input text", [Expected speakers to be extracted by TTC])
    """
    content = (path / file_name).read_text(encoding="utf-8").split(delimiter)
    return "\n" + content[0], [s for s in content[1].split("\n") if s.strip()]
