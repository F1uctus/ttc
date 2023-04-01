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

    assert "Сзету" == play.last_speaker.text


def test_noun_with_aux(cc):
    text = (
        "– Что это такое? – Ее голос был тихим, как шепот. – Ты можешь мне показать."
        " Я никому не расскажу. Это сокровище? Ты отрезал кусочек от покрывала"
        " ночи и спрятал его? Это сердце жука – маленькое, но сильное?"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Что это такое?",
        "Ты можешь мне показать."
        " Я никому не расскажу. Это сокровище? Ты отрезал кусочек от покрывала"
        " ночи и спрятал его? Это сердце жука – маленькое, но сильное?",
    ]
    play = cc.connect_play(dialogue)
    actual_speakers = [str(s.lemma_) for s in play.speakers if s]
    assert actual_speakers == [
        "она голос",  # ее --lemma-> она
        "она голос",
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
    actual_speakers = [str(s.lemma_) for s in play.speakers if s]
    assert actual_speakers == [
        "моаш",
        "одноглазый коротышка-сержант",
    ]


@pytest.mark.xfail(reason="misprediction of dative as feminine", raises=AssertionError)
def test_ignorance_of_dative(cc):
    text = (
        "– Что происходит? – спросил Калак.\n"
        "– На этот раз погиб только один. – Низкий голос был спокоен.\n"
        "– Таленель. – Не хватало лишь Клинка Чести, который принадлежал Таленелю."
        #                              ^------ problematic places -------^
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Что происходит?",
        "На этот раз погиб только один.",
        "Таленель.",
    ]
    play = cc.connect_play(dialogue)
    actual_speakers = [str(s.lemma_) for s in play.speakers if s]
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
    assert [str(s).lower() for s in expected_speakers] == [
        str(s).lower() for s in play.speakers if s
    ]
    assert expected_replicas == list(map(str, play.replicas))
