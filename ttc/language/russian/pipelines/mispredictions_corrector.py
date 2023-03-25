from spacy import Language
from spacy.symbols import PART, VERB  # type: ignore
from spacy.tokens import Doc

NAME = "correct_mispredictions"

MISPREDICTED_VERBS_SENT_START = {
    "беги",
    "поведай",
    "засветись",
    "забудь",
}

MISPREDICTED_VERBS = {
    "бормочет",
}

PARTICLES = {
    "а",
    "авось",
    "ах",
    "бишь",
    "благо",
    "более",
    "больше",
    "будто",
    "буквально",
    "бы",
    "бывает",
    "бывало",
    "было",
    "ведь",
    "во",
    "вовсе",
    "вон",
    "вот",
    "вроде",
    "вряд",
    "все",
    "всего",
    "всё",
    "где",
    "да",
    "давай",
    "даже",
    "дай",
    "далеко",
    "де",
    "действительно",
    "дескать",
    "едва",
    "единственно",
    "если",
    "ещё",
    "же",
    "знай",
    "и",
    "или",
    "именно",
    "ин",
    "исключительно",
    "ишь",
    "как",
    "какое",
    "кое",
    "куда",
    "ладно",
    "ли",
    "лишь",
    "лучше",
    "навряд",
    "небось",
    "неужели",
    "нехай",
    "нечего",
    "ни",
    "ничего",
    "но",
    "ну",
    "о",
    "однако",
    "окончательно",
    "оно",
    "отнюдь",
    "ох",
    "подлинно",
    "положительно",
    "почти",
    "просто",
    "прямо",
    "пускай",
    "пусть",
    "разве",
    "решительно",
    "ровно",
    "самое",
    "себе",
    "скорее",
    "словно",
    "смотри",
    "совершенно",
    "совсем",
    "спасибо",
    "сём",
    "так",
    "таки",
    "там",
    "тебе",
    "то",
    "тоже",
    "только",
    "точно",
    "уж",
    "ух",
    "хорошо",
    "хоть",
    "чай",
    "чего",
    "что",
    "чтоб",
    "чтобы",
    "эк",
    "это",
    "эх",
}
PARTICLE_ENDINGS = {" ли", " ль", "-то", "-тка", "-де", "-ка", "-точь", "-с"}


# TODO: Не хватало лишь Клинка Чести.
#          -------      not fem
#          ^ deduce from that


def nearest_linked_verb(t):
    head = t.head
    while head and head.pos != VERB:
        if head == head.head:
            return None
        head = head.head
    else:
        return head


@Language.component(NAME)
def _correct_mispredictions(doc: Doc):
    for token in doc:

        norm_text = token.text.strip().lower()

        if norm_text in MISPREDICTED_VERBS:
            token.pos_ = "VERB"
        if token.is_sent_start and norm_text in MISPREDICTED_VERBS_SENT_START:
            token.pos_ = "VERB"

        if token.is_title and token.pos == PART and token.lower_ not in PARTICLES:
            if any(token.lower_.endswith(e) for e in PARTICLE_ENDINGS):
                continue
            if nearest_linked_verb(token):
                token.pos_ = "PROPN"

    return doc
