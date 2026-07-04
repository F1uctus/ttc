from pathlib import Path

from ttc.corpus import (
    DELIMITER,
    UNATTRIBUTED,
    canonical_actor,
    find_corpus_files,
    load_corpus_file,
    normalize_name,
    parse_corpus_content,
    serialize_corpus_file,
)

TEXTS_PATH = Path(__file__).parent / "russian" / "texts"


def test_normalize_name():
    assert normalize_name("  Ясна ") == "ясна"
    assert normalize_name("Один  из\tшахтеров") == "один из шахтеров"
    assert normalize_name("Её светлость") == "ее светлость"


def test_canonical_actor_unattributed():
    assert canonical_actor(None, {}) == UNATTRIBUTED
    assert canonical_actor("None", {}) == UNATTRIBUTED
    assert canonical_actor("none", {}) == UNATTRIBUTED


def test_canonical_actor_aliases():
    aliases = {"принцесса": "ясна", "светлость": "ясна"}
    assert canonical_actor("Принцесса", aliases) == "ясна"
    assert canonical_actor("ясна", aliases) == "ясна"
    assert canonical_actor("Тозбек", aliases) == "тозбек"


def test_parse_two_sections():
    content = (
        "Текст истории.\n"
        f"{DELIMITER}\n"
        "Ясна::Привет.\n"
        "Шаллан::Светлость…\n"
    )
    cf = parse_corpus_content(content)
    assert cf.text == "Текст истории."
    assert cf.pairs == [("Ясна", "Привет."), ("Шаллан", "Светлость…")]
    assert cf.aliases == {}


def test_parse_replica_containing_double_colon():
    content = f"Текст.\n{DELIMITER}\nА::Смотри:: вот.\n"
    cf = parse_corpus_content(content)
    assert cf.pairs == [("А", "Смотри:: вот.")]


def test_parse_alias_section():
    content = (
        "Текст.\n"
        f"{DELIMITER}\n"
        "принцесса::Да.\n"
        f"{DELIMITER}\n"
        "# кто есть кто\n"
        "Ясна = принцесса | её светлость\n"
    )
    cf = parse_corpus_content(content)
    assert cf.aliases == {"принцесса": "ясна", "ее светлость": "ясна"}
    assert canonical_actor(cf.pairs[0][0], cf.aliases) == "ясна"


def test_round_trip():
    text = "Первая строка.\nВторая строка."
    pairs = [("Ясна", "Да."), ("None", "Кто здесь?")]
    aliases = {"Ясна": ["принцесса", "светлость"]}
    content = serialize_corpus_file(text, pairs, aliases)
    cf = parse_corpus_content(content)
    assert cf.text == text
    assert cf.pairs == pairs
    assert cf.aliases == {"принцесса": "ясна", "светлость": "ясна"}
    # serialization is stable
    assert serialize_corpus_file(cf.text, cf.pairs, {"Ясна": ["принцесса", "светлость"]}) == content


def test_existing_corpus_parses():
    files = find_corpus_files(TEXTS_PATH)
    assert files, "no corpus files found"
    total_pairs = 0
    for f in files:
        cf = load_corpus_file(f)
        assert cf.text, f"{f} has empty text"
        assert cf.pairs, f"{f} has no annotated pairs"
        for actor, replica in cf.pairs:
            assert actor and replica, f"{f} has a malformed pair: {(actor, replica)!r}"
        total_pairs += len(cf.pairs)
    assert total_pairs >= 381
