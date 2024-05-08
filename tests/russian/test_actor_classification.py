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


def normalize(spans):
    return [str(s.lemma_) for s in spans if s]


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
    assert normalize(play.actors) == [
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
    assert normalize(play.actors) == [
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
    assert normalize(play.actors) == [
        "калак",
        "низкий голос",
        "калак",
    ]


def test_preceding_line_without_author_punct_with_one_sent(cc):
    text = (
        "– Понял, – пробормотал общительный раб, – а за что ты получил первое клеймо?\n"
        "Каладин выпрямился, чувствуя, как что-то постукивает под днищем фургона.\n"
        "– Я убил светлоглазого."
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Понял,",
        "а за что ты получил первое клеймо?",
        "Я убил светлоглазого.",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "общительный раб",
        "общительный раб",
        "каладин",
    ]

    text = (
        "– Ваши атрибуты, моя госпожа. – Ялб сделал ударение на «и».\n"
        "Шаллан вскинула бровь.\n"
        "– Мои… атрибуты?"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Ваши атрибуты, моя госпожа.",
        "Мои… атрибуты?",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "ялб",
        "шаллан",
    ]


@pytest.mark.xfail(reason="This requires replica analysis", raises=AssertionError)
def test_preceding_line_without_author_punct_with_multiple_sents(cc):
    text = (
        "Парень закрыл глаза и уткнулся лбом в прутья клетки:\n"
        "– Я так устал…\n"
        "Он не имел в виду физическое изнеможение. Каладин просто чувствовал… усталость. Он так устал…\n"
        #                                          (*)
        "– Ты и раньше уставал.\n"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Я так устал…",
        "Ты и раньше уставал.",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "парень",
        "none",  # FIXME: Now this is (*)
    ]


def test_unannotated_alternation(cc):
    text = (
        "Каладин отвернулся. Продолжая сидеть, он положил руку поперек прутьев.\n"
        "– Ну? – спросил раб.\n"
        "– Ты дурак. Будешь отдавать мне половину своей еды – слишком ослабеешь.\n"
        "– Но…"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Ну?",
        "Ты дурак. Будешь отдавать мне половину своей еды – слишком ослабеешь.",
        "Но…",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "раб",
        "каладин",
        "раб",
    ]


def test_allow_sparse_repetition(cc):
    text = (
        "«Не дури», – приказал себе Каладин.\n"
        "Блат подтащил раба к костру. Каладин почувствовал облегчение. «Ну вот, – подумал он. – Возможно, ты еще можешь кому-то помочь»"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Не дури",
        "Ну вот,",
        "Возможно, ты еще можешь кому-то помочь",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "каладин",
        "каладин",
        "каладин",
    ]


def test_allow_actor_in_acl_relcl(cc):
    text = (
        "Вдвоем они направились по коридору к той двери, откуда вышла Ясна.\n"
        "– Отец? – окликнула она. – Ты что-то от меня скрываешь?"
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Отец?",
        "Ты что-то от меня скрываешь?",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "ясна",
        "ясна",
    ]

    text = (
        "– Это что? – Разбойник вытащил камень из ладони того, который считал добычу."
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Это что?",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "разбойник",
    ]


def test_verb_only_reference(cc):
    text = (
        "Йезриен поднял меч и вонзил его в камень. Ненадолго застыл, а потом склонил голову и отвернулся, словно его одолел стыд:\n"
        "– Мы по собственной воле взвалили это бремя на свои плечи."
    )
    dialogue = cc.extract_dialogue(text)
    assert list(map(str, dialogue.replicas)) == [
        "Мы по собственной воле взвалили это бремя на свои плечи.",
    ]
    play = cc.connect_play(dialogue)
    assert normalize(play.actors) == [
        "йезриен",
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
