from pathlib import Path
from typing import Final

import pytest

import ttc
from tests.test_text_helpers import find_test_texts, load_test

TEXTS_PATH: Final = Path(__file__).parent / "texts"


@pytest.fixture
def cc():
    yield ttc.load("ru")


def test_extract_dialogue(cc):
    dialogue = cc.extract_dialogue(
        """\
— Что есть счастье? — вдруг громко спрашивает Гриша. Михайло смотрит на него, и вся физиономия его расплывается, как от сна. Всего полчаса назад он, отобрав четырех малышей шестилетнего возраста, лихо отплясывал с ними в хороводе, покуда не упал, чуть не раздавив одного из них.
Не получив ответа, Гриша жадно макает свою кудрявую голову в пиво, потом нагибается к Михайле, хлопает его по колену и хрипло говорит:
— Слышь, браток... Почему ты счастлив... Скажи... Корову подарю...
Михайло важно снимает огромную Гришину ручищу с колен и отвечает:
— Ты меня не трожь.
Гриша вздыхает.
— Ведь все вроде у меня есть, что у тебя... Корова, четыре бабы, хата с крышей, пчелы... Подумаю так: чево мне яще желать? Ничевошеньки. А автомобиля: «ЗИЛ» там или грузовик — мне и задаром не нужно: тише едешь, дальше будешь... Все у меня есть, — заключает Гриша.
Михайло молчит, утонув в пиве.
— Только мелочное все это, что у меня есть, — продолжает Гриша. — Не по размерам, а просто так, по душе... Мелочное, потому что мысли у меня есть. Оттого и страшно.
— Иди ты, — отвечает Михайло.
— Тоскливо мне чего-то жить, Мишук, — бормочет Гриша, опустив свою квадратную челюсть на стол.
— А чево?
— Да так... Тяжело все... Люди везде, комары... Опять же ночи... Облака... Очень 
скушно мне вставать по утрам... Руки... Сердце...
— Плохое это, — мычит Михайло.
Напившись пива, он становится разговорчивей, но так и не поднимая полностью завесы над своей великой тайной — тайной счастья. Лишь жирное, прыщеватое лицо его сияет как масленое солнышко.
— К бабе, к примеру, подход нужен, — поучает он, накрошив хлеба в рот. — Баба, она не корова, хоть и пузо у нее мягкое... Ее с замыслом выбирать нужно... К примеру, у меня есть девки на все случаи: одна, с которой я сплю завсегда после грозы, другая лунная (при луне, значит), с третьей — я только после баньки... Вот так.
Михайло совсем растаял от счастья и опять утонул в пиве.
— А меня все это не шевелит, — рассуждает Гриша. — Я и сам все это знаю.
— Счастье — это довольство... И чтоб никаких мыслей, — наконец проговаривается Михайло.
— Вот мыслей-то я и боюсь, — обрадовался Гриша. — Завсегда они у меня скачут. Удержу нет. И откуда только они появляются. Намедни совсем веселый был. Хотя и дочка кипятком обварилась. Шел себе просто по дороге, свистел. И увидал елочку, махонькую такую, облеванную... И так чего-то пужливо мне стало, пужливо... Или вот когда просто мысль появляется... Все ничего, ничего, пусто, и вдруг — бац! — мысль... Боязно очень. Особенно о себе боюсь думать.
— Ишь ты... О себе — оно иной раз бывает самое приятное думать, — скалится Михайло, поглаживая себя по животу.
В деревушке, как в лесу, не слышно ни единого непристойного звука. Все спит. Лишь вдали, поводя бедрами, выходит посмотреть на тучки упитанная дева, Тамарочка.
"""
    )
    assert True


@pytest.mark.parametrize("file_name", find_test_texts(TEXTS_PATH))
def test_text_to_play(cc, file_name):
    text, expected_speakers = load_test(TEXTS_PATH, file_name)
    play = cc.connect_play(cc.extract_dialogue(text))
    actual_speakers = [str(spk.text) for spk in play.content.values()]
    print("\n" * 2 + "\n".join(str(spk) for spk in play.content.values()))
    assert expected_speakers == actual_speakers


def test_name_reference(cc):
    play = cc.connect_play(
        cc.extract_dialogue(
            """
– …и вот тогда он поклялся служить мне, – завершил Тук. – И с той поры со мной.
Слушатели повернулись к Сзету.
– Это правда, – подтвердил он, как было приказано заранее. – До последнего слова.
"""
        )
    )

    assert "сзету" == list(play.content.values())[-1].first_token.lemma_
