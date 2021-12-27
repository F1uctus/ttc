from pathlib import Path
from pprint import pprint
from typing import Final

import pytest

import ttc
from tests.test_text_helpers import find_test_texts, load_test

TEXTS_PATH: Final = Path(__file__).parent / "texts"


@pytest.fixture
def cc():
    yield ttc.load("ru")


def test_single_pre_replica_by_pushkin(cc):
    text = "«Далече ли до крепости?» – спросил я у своего ямщика"
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 1
    assert str(dialogue.replicas[0]) == "Далече ли до крепости?"


def test_single_pre_replica_by_mamleyev(cc):
    text = "— Что есть счастье? — вдруг громко спрашивает Гриша."
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 1
    assert str(dialogue.replicas[0]) == "Что есть счастье?"


def test_single_post_replica_by_pushkin(cc):
    text = "Старый священник подошел ко мне с вопросом: «Прикажете начинать?»"
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 1
    assert str(dialogue.replicas[0]) == "Прикажете начинать?"


def test_raw_replica_with_intervention_by_pushkin(cc):
    text = """
    «Тише, – говорит она мне, – отец болен, при смерти, и желает с тобою проститься»
    """
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 2
    assert str(dialogue.replicas[0]) == "Тише,"
    assert (
        str(dialogue.replicas[1])
        == "отец болен, при смерти, и желает с тобою проститься"
    )


def test_replica_with_intervention_containing_a_dash_by_sanderson(cc):
    text = (
        "– Не-а. Они все так выглядят. Эй, а это что такое? "
        "– Разбойник вытащил искрящийся камень размером со сферу из ладони того, "
        "который считал добычу. Камень выглядел заурядно – просто кусочек "
        "скалы с несколькими кристаллами кварца и ржавой железной жилой. "
        "– Это что?"
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 2
    assert (
        str(dialogue.replicas[0]) == "Не-а. Они все так выглядят. Эй, а это что такое?"
    )
    assert str(dialogue.replicas[1]) == "Это что?"


def test_replica_with_intervention_containing_a_comma_and_dash_by_sanderson(cc):
    text = (
        "– Нет! – рявкнул Каладин. – Вылазки с мостом выматывают нас. О, я знаю, "
        "нас заставляют работать – осматривать ущелья, чистить нужники, "
        "драить полы. Но солдаты не хотят, чтобы мы по-настоящему трудились, "
        # problematic place
        "– им просто нужно нас занять. "
        "Поручая нам какое-нибудь дело, они про нас забывают. "
        "Поскольку я теперь ваш старшина, моя задача – сохранить вам жизнь. "
        "Стрелы паршенди никуда не исчезнут, поэтому я буду менять вас. "
        "Хочу сделать вас сильнее, чтобы на последнем отрезке пути с мостом, "
        "когда полетят стрелы, вы смогли бежать быстро. "
        "– Он посмотрел в глаза каждому. "
        "– Я собираюсь устроить так, чтобы Четвертый мост больше не потерял "
        "ни одного человека."
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 3
    assert (
        str(dialogue.replicas[-1]) == "Я собираюсь устроить так, чтобы Четвертый мост"
        " больше не потерял ни одного человека."
    )


@pytest.mark.parametrize("file_name", find_test_texts(TEXTS_PATH))
def test_text_to_play(cc, file_name):
    text, expected_speakers = load_test(TEXTS_PATH, file_name)
    print()
    dialogue = cc.extract_dialogue(text)
    pprint(dialogue)
    # play = cc.connect_play(dialogue)
    # actual_speakers = [str(spk.text) for spk in play.content.values()]
    # pprint(play.content)
    # assert expected_speakers == actual_speakers


def test_name_reference(cc):
    dialogue = cc.extract_dialogue(
        """
– …и вот тогда он поклялся служить мне, – завершил Тук. – И с той поры со мной.
Слушатели повернулись к Сзету.
– Это правда, – подтвердил он, как было приказано заранее. – До последнего слова.
"""
    )
    play = cc.connect_play(dialogue)

    assert "сзету" == list(play.content.values())[-1].first_token.lemma_
