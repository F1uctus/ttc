# encoding: utf-8
from pathlib import Path

from ttc import extract_speakers

texts = []
for entry in (Path(".") / "language" / "russian" / "texts").iterdir():
    if entry.is_file() and entry.suffix == ".txt":
        texts.append("\n" + entry.read_text() + "\n")

# TODO Достоевский, множество персонажей, описания персонажей, графы связей
rs = extract_speakers(texts[-1], "ru")

print()
print("\n".join(replica.to_string() for replica in rs))
print([r.speaker for r in rs])
