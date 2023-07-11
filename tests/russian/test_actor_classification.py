from pathlib import Path
from typing import List, Tuple, Final

import pytest

import ttc

TEXTS_PATH: Final = Path(__file__).parent / "texts"


def find_test_texts(path: Path):
    return [e.name for e in path.iterdir() if e.suffix == ".txt"]


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

    assert "Сзету" == play.last_actor.text


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
    actual_actors = [str(s.lemma_) for s in play.actors if s]
    assert actual_actors == [
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
    actual_actors = [str(s.lemma_) for s in play.actors if s]
    assert actual_actors == [
        "моаш",
        "одноглазый коротышка-сержант",
    ]


def test_ignorance_of_impersonal(cc):
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
    actual_actors = [str(s.lemma_) for s in play.actors if s]
    assert actual_actors == [
        "калак",
        "низкий голос",
        "калак",
    ]


def test_context_without_punct(cc):
    text = (
        "– Понял, – пробормотал общительный раб, – видимо, стоит задать вопрос по-другому. За что ты получил первое клеймо?\n"
        "Каладин выпрямился, чувствуя, как что-то постукивает под днищем катящегося фургона.\n"
        "– Я убил светлоглазого."
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Понял,",
        "видимо, стоит задать вопрос по-другому. За что ты получил первое клеймо?",
        "Я убил светлоглазого.",
    ]
    play = cc.connect_play(dialogue)
    actual_actors = [str(s.lemma_) for s in play.actors if s]
    assert actual_actors == [
        "общительный раб",
        "общительный раб",
        "каладин",
    ]


def test_unannotated_alternation(cc):
    text = (
        "Каладин отвернулся, всматриваясь в бесконечные холмы и беспокойную, подвижную траву. Продолжая сидеть, свесив ноги наружу, он положил руку поперек прутьев клетки и уперся в нее лбом.\n"
        "– Ну? – спросил раб.\n"
        "– Ты дурак. Будешь отдавать мне половину своей еды – слишком ослабеешь к моменту побега, если я таковой устрою. А я не устрою. Ничего не получится.\n"
        "– Но…"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Ну?",
        "Ты дурак. Будешь отдавать мне половину своей еды – слишком ослабеешь к моменту побега, если я таковой устрою. А я не устрою. Ничего не получится.",
        "Но…",
    ]
    play = cc.connect_play(dialogue)
    actual_actors = [str(s.lemma_) for s in play.actors if s]
    assert actual_actors == [
        "раб",
        "каладин",
        "раб",
    ]


def test_allow_sparse_repetition(cc):
    text = (
        "«Не дури», – приказал себе Каладин.\n"
        "Дверь клетки вернулась на прежнее место, щелкнул замок. Клетки необходимы – Твлакву приходилось защищать свое хрупкое имущество от Великих бурь.\n"
        "Блат подтащил раба к костру, к непочатому бочонку с водой. Каладин почувствовал облегчение. «Ну вот, – подумал он. – Возможно, ты еще можешь кому-то помочь»"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Не дури",
        "Ну вот,",
        "Возможно, ты еще можешь кому-то помочь",
    ]
    play = cc.connect_play(dialogue)
    actual_actors = [str(s.lemma_) for s in play.actors if s]
    assert actual_actors == [
        "каладин",
        "каладин",
        "каладин",
    ]


@pytest.mark.xfail(reason="some test files are still failing", raises=AssertionError)
@pytest.mark.parametrize("file_name", find_test_texts(TEXTS_PATH))
def test_text_to_play(cc, file_name):
    text, expected_result = load_test(TEXTS_PATH, file_name)
    expected_replicas, expected_actors = expected_result
    dialogue = cc.extract_dialogue(text)
    play = cc.connect_play(dialogue)
    print("\n", play, sep="")
    expected_play_rels = {a: b for a, b in zip(expected_replicas, expected_actors)}
    actual_play_rels = {str(r): str(s).lower() for r, s in play.lines}
    assert expected_play_rels == actual_play_rels
