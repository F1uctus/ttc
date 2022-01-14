from pathlib import Path
from pprint import pprint
from typing import Final

import pytest

import ttc

TEXTS_PATH: Final = Path(__file__).parent / "texts"


def find_test_texts(path: Path):
    return [e.name for e in path.iterdir() if e.suffix == ".txt"]


def load_test(path: Path, file_name: str, delimiter="-" * 20):
    """
    :returns: ("Input text", [Expected speakers to be extracted by TTC])
    """
    content = (path / file_name).read_text(encoding="utf-8").split(delimiter)
    return "\n" + content[0], [s for s in content[1].split("\n") if s.strip()]


@pytest.fixture
def cc():
    yield ttc.load("ru")


def test_name_reference(cc):
    dialogue = cc.extract_dialogue(
        """
– …и вот тогда он поклялся служить мне, – завершил Тук. – И с той поры со мной.
Слушатели повернулись к Сзету.
– Это правда, – подтвердил он, как было приказано заранее. – До последнего слова.
"""
    )
    play = cc.connect_play(dialogue)

    assert "к Сзету" == list(play.content.values())[-1].text


@pytest.mark.parametrize("file_name", find_test_texts(TEXTS_PATH))
def test_text_to_play(cc, file_name):
    text, expected_speakers = load_test(TEXTS_PATH, file_name)
    print()
    dialogue = cc.extract_dialogue(text)
    pprint(dialogue)
    play = cc.connect_play(dialogue)
    actual_speakers = [str(spk.lemma) for spk in play.content.values()]
    pprint(play.content)
    assert expected_speakers == actual_speakers
