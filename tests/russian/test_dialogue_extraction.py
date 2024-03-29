import pytest

import ttc


@pytest.fixture(scope="module")
def cc():
    yield ttc.load("ru")


def test_single_pre_replica_by_pushkin(cc):
    text = "«Далече ли до крепости?» – спросил я у своего ямщика"
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == ["Далече ли до крепости?"]


def test_single_pre_replica_by_mamleyev(cc):
    text = "— Что есть счастье? — вдруг громко спрашивает Гриша."
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == ["Что есть счастье?"]


def test_single_post_replica_by_pushkin(cc):
    text = "Старый священник подошел ко мне с вопросом: «Прикажете начинать?»"
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == ["Прикажете начинать?"]


def test_raw_replica_with_intervention_by_pushkin(cc):
    text = (
        "«Тише, – говорит она мне, – отец"
        " болен, при смерти, и желает с тобою проститься»"
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Тише,",
        "отец болен, при смерти, и желает с тобою проститься",
    ]


def test_replica_with_intervention_containing_a_hyphen_by_sanderson(cc):
    text = (
        "– Не-а. Они все так выглядят. Эй, а это что такое? "
        "– Разбойник вытащил искрящийся камень размером со сферу из ладони того, "
        "который считал добычу. Камень выглядел заурядно – просто кусочек "
        "скалы с несколькими кристаллами кварца и ржавой железной жилой. "
        "– Это что?"
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Не-а. Они все так выглядят. Эй, а это что такое?",
        "Это что?",
    ]


def test_replica_with_complex_sentence_inside(cc):
    text = (
        "Джон продолжил:\n"
        "— Делал ли что-нибудь для этого Штольц, что делал "
        "и как делал, — мы этого не знаем."
        #           ^^^^ problematic place
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Делал ли что-нибудь для этого Штольц, что делал "
        "и как делал, — мы этого не знаем.",
    ]


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
    d = cc.extract_dialogue(text)
    assert len(d.replicas) == 3
    assert str(d.replicas[1]) == (
        "Вылазки с мостом выматывают нас. О, я знаю, "
        "нас заставляют работать – осматривать ущелья, чистить нужники, "
        "драить полы. Но солдаты не хотят, чтобы мы по-настоящему трудились, "
        "– им просто нужно нас занять. "
        "Поручая нам какое-нибудь дело, они про нас забывают. "
        "Хочу сделать вас сильнее, чтобы на последнем отрезке пути с мостом, "
        "когда полетят стрелы, вы смогли бежать быстро."
    )
    assert str(d.replicas[2]) == (
        "Я собираюсь устроить так, чтобы Четвертый мост"
        " больше не потерял ни одного человека."
    )


def test_replica_with_intervention_starting_with_a_comma_and_hyphen_by_sanderson(cc):
    text = (
        "– Ага, – согласился Лейтен, высокий"
        " крепыш с курчавыми волосами. – Это точно."
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Ага,",
        "Это точно.",
    ]


def test_replica_ending_with_a_comma_and_hyphen_by_mamleyev(cc):
    text = (
        "— Счастье — это довольство... И чтоб никаких "
        "мыслей, — наконец проговаривается Михайло."
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Счастье — это довольство... И чтоб никаких мыслей,",
    ]


def test_replica_with_onomatopoeia_by_mamleyev(cc):
    text = (
        "— Вот их я и боюсь, — обрадовался Гриша. "
        "— Все пусто, и вдруг — бац! "
        # <<< problematic place >>>
        "— мысль... Боязно очень."
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Вот их я и боюсь,",
        "Все пусто, и вдруг — бац! — мысль... Боязно очень.",
    ]


def test_replica_with_repetitive_onomatopoeia(cc):
    text = "— Все тихо, и вдруг — бам-бам-бам! — заколотили в дверь..."
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Все тихо, и вдруг — бам-бам-бам! — заколотили в дверь...",
    ]


def test_false_quoted_piece_with_trailing_hyphen(cc):
    text = (
        "Она – Каладин вдруг понял, что воспринимает"
        " спрена ветра как «ее», – носила простое платье."
        # problematic piece ^^
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == []


def test_quoted_insertion_with_no_action_verb(cc):
    text = (
        "«Постоянный кашель, – всплыло в памяти Каладина,"
        " – сопровождаемый избытком мокроты и ночным"
        " лихорадочным бредом. Скорее всего, кашель-скрежетун»."
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Постоянный кашель,",
        "сопровождаемый избытком мокроты и ночным лихорадочным бредом. Скорее всего, кашель-скрежетун",
    ]


def test_unquoted_insertion_with_no_action_verb(cc):
    text = (
        "— Он умер?! — Мальчика так потрясла эта новость, что он забыл про свой бок."
        " Уистиоу всегда был где-то рядом и просто не мог исчезнуть."
        " А что же будет с Лараль?"
        " — С ним же все было в порядке на той неделе!"
    )
    d = cc.extract_dialogue(text)
    assert list(map(str, d.replicas)) == [
        "Он умер?!",
        "С ним же все было в порядке на той неделе!",
    ]
