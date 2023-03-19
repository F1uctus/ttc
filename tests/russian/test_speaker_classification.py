from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Final

import pytest

import ttc

TEXTS_PATH: Final = Path(__file__).parent / "texts"


def find_test_texts(path: Path):
    return [e.name for e in path.iterdir() if e.suffix == ".txt"]
    # return [[e.name for e in path.iterdir() if "dekret" in str(e)][0]]


def parse_conversation(text: str) -> Tuple[List[str], List[str]]:
    lines = [l for line in text.split("\n") if (l := line.strip())]
    dialogue = [(r, s.lower()) for s, r in [line.split("::") for line in lines]]
    return tuple(list(x) for x in zip(*dialogue))


def load_test(
    path: Path, file_name: str, delimiter="-" * 20
) -> Tuple[str, Tuple[List[str], List[str]]]:
    """
    Returns
    -------
    (Input text, Expected conversation to be extracted by ttc.)
    """
    content = (path / file_name).read_text(encoding="utf-8").split(delimiter)
    return content[0].strip(), parse_conversation(content[1].strip())


@pytest.fixture(scope="module")
def cc():
    yield ttc.load("ru")


def test_name_reference(cc):
    dialogue = cc.extract_dialogue(
        "– …и вот тогда он поклялся служить мне, – завершил"
        " Тук. – И с той поры со мной.\n"
        "Слушатели повернулись к Сзету.\n"
        "– Это правда, – подтвердил он, как"
        " было приказано заранее. – До последнего слова.\n"
    )
    play = cc.connect_play(dialogue)

    assert "к Сзету" == list(play.content.values())[-1].text


def test_referential_pronoun_by_tolstoy(cc):
    text = (
        "– Он умрет. Третье, – что бишь еще ты сказал? – Князь Андрей загнул третий палец.\n"
        "- Ничего не говорил!\n"
        "– У тебя лишний работник пропал! – сказал он, отвернувшись от Пьера.\n"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Он умрет. Третье, – что бишь еще ты сказал?",
        "Ничего не говорил!",
        "У тебя лишний работник пропал!",
    ]
    play = cc.connect_play(dialogue)
    actual_speakers = [
        str(spk.lemma_) if (spk := play.content.get(r)) else None
        for r in dialogue.replicas
    ]
    assert actual_speakers == [
        "князь андрей",
        None,
        "князь андрей",
    ]


def test_hyphenated_noun_chunk(cc):
    text = (
        "– Эй, Газ, – позвал Моаш, приложив ладони рупором ко рту.\n"
        "Одноглазый коротышка-сержант разговаривал с солдатами неподалеку.\n"
        "– Чего надо? – Тот скорчил недовольную мину."
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Эй, Газ,",
        "Чего надо?",
    ]
    play = cc.connect_play(dialogue)
    actual_speakers = [str(spk.lemma_) for spk in play.content.values() if spk]
    assert actual_speakers == [
        "моаш",
        "одноглазый коротышка-сержант",
    ]


def test_ignorance_of_dative(cc):
    text = (
        "– Что происходит? – спросил Калак.\n"
        "– На этот раз погиб только один. – Низкий голос был спокоен.\n"
        "– Таленель. – Не хватало лишь Клинка Чести, который принадлежал Таленелю."
        #                                             problematic place -^
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Что происходит?",
        "На этот раз погиб только один.",
        "Таленель.",
    ]
    play = cc.connect_play(dialogue)
    actual_speakers = [str(spk.lemma_) for spk in play.content.values() if spk]
    assert actual_speakers == [
        "калак",
        "низкий голос",
        "калак",
    ]


@pytest.mark.xfail(reason="some test files are still failing", raises=AssertionError)
@pytest.mark.parametrize("file_name", find_test_texts(TEXTS_PATH))
def test_text_to_play(cc, file_name):
    text, expected_result = load_test(TEXTS_PATH, file_name)
    expected_replicas, expected_speakers = expected_result
    dialogue = cc.extract_dialogue(text)
    play = cc.connect_play(dialogue)
    print()
    print(play)
    print("\n")
    assert [cc.language(s)[:].lemma_ for s in expected_speakers] == [
        s.lemma_ for s in play.content.values()
    ]
    assert expected_replicas == list(map(str, play.content.keys()))
