import pytest

import ttc


@pytest.fixture(scope="module")
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
    assert str(dialogue.replicas[1]) == (
        "отец болен, при смерти, и желает с тобою проститься"
    )


def test_replica_with_intervention_containing_a_hyphen_by_sanderson(cc):
    text = (
        "– Не-а. Они все так выглядят. Эй, а это что такое? "
        "– Разбойник вытащил искрящийся камень размером со сферу из ладони того, "
        "который считал добычу. Камень выглядел заурядно – просто кусочек "
        "скалы с несколькими кристаллами кварца и ржавой железной жилой. "
        "– Это что?"
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 2
    assert str(dialogue.replicas[0]) == (
        "Не-а. Они все так выглядят. Эй, а это что такое?"
    )
    assert str(dialogue.replicas[1]) == "Это что?"


def test_replica_with_complex_sentence_inside(cc):
    text = (
        "Джон продолжил:\n"
        "— Делал ли что-нибудь для этого Штольц, что делал "
        "и как делал, — мы этого не знаем."
        #           ^^^^ problematic place
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 1
    assert str(dialogue.replicas[0]) == (
        "Делал ли что-нибудь для этого Штольц, что делал "
        "и как делал, — мы этого не знаем."
    )


def test_replica_with_intervention_containing_a_comma_and_hyphen_by_sanderson(cc):
    text = (
        "– Нет! – рявкнул Каладин. – Вылазки с мостом выматывают нас. О, я знаю, "
        "нас заставляют работать – осматривать ущелья, чистить нужники, "
        "драить полы. Но солдаты не хотят, чтобы мы по-настоящему трудились, "
        # <<< problematic place >>>
        "– им просто нужно нас занять. "
        "Поручая нам какое-нибудь дело, они про нас забывают. "
        "Хочу сделать вас сильнее, чтобы на последнем отрезке пути с мостом, "
        "когда полетят стрелы, вы смогли бежать быстро. "
        "– Он посмотрел в глаза каждому. "
        "– Я собираюсь устроить так, чтобы Четвертый мост больше не потерял "
        "ни одного человека."
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 3
    assert str(dialogue.replicas[1]) == (
        "Вылазки с мостом выматывают нас. О, я знаю, "
        "нас заставляют работать – осматривать ущелья, чистить нужники, "
        "драить полы. Но солдаты не хотят, чтобы мы по-настоящему трудились, "
        "– им просто нужно нас занять. "
        "Поручая нам какое-нибудь дело, они про нас забывают. "
        "Хочу сделать вас сильнее, чтобы на последнем отрезке пути с мостом, "
        "когда полетят стрелы, вы смогли бежать быстро."
    )
    assert str(dialogue.replicas[2]) == (
        "Я собираюсь устроить так, чтобы Четвертый мост"
        " больше не потерял ни одного человека."
    )


def test_replica_with_intervention_starting_with_a_comma_and_hyphen_by_sanderson(cc):
    text = (
        "– Ага, – согласился Лейтен, высокий крепыш с курчавыми волосами. – Это точно."
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 2
    assert str(dialogue.replicas[0]) == "Ага,"
    assert str(dialogue.replicas[1]) == "Это точно."


def test_replica_ending_with_a_comma_and_hyphen_by_mamleev(cc):
    text = (
        "— Счастье — это довольство... И чтоб никаких "
        "мыслей, — наконец проговаривается Михайло."
    )
    dialogue = cc.extract_dialogue(text)
    assert len(dialogue.replicas) == 1
    assert str(dialogue.replicas[0]) == (
        "Счастье — это довольство... И чтоб никаких мыслей,"
    )
