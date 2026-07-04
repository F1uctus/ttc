from pathlib import Path

from ttc.corpora.schema import (
    Character,
    CorpusDoc,
    Cue,
    Mention,
    Replica,
    doc_from_dict,
    read_jsonl,
    to_dict,
    validate,
    write_jsonl,
)


def make_doc() -> CorpusDoc:
    return CorpusDoc(
        doc_id="pdnc/emma/1",
        lang="en",
        domain="prose",
        source="pdnc",
        license="CC-BY-NC-4.0",
        text="“Come here,” said Emma to Harriet.",
        replicas=[
            Replica(0, 12, speaker="char_0", addressee="char_1",
                    qtype="explicit", cue=Cue(13, 22))
        ],
        characters=[
            Character("char_0", "Emma", aliases=["Miss Woodhouse"], gender="f"),
            Character("char_1", "Harriet"),
        ],
        mentions=[Mention(18, 22, "char_0")],
    )


def test_round_trip_dict():
    doc = make_doc()
    assert doc_from_dict(to_dict(doc)) == doc


def test_round_trip_jsonl(tmp_path: Path):
    docs = [make_doc(), make_doc()]
    path = tmp_path / "c.jsonl"
    write_jsonl(docs, path)
    assert list(read_jsonl(path)) == docs


def test_none_cue_and_speaker_survive_round_trip():
    doc = make_doc()
    doc.replicas[0].cue = None
    doc.replicas[0].speaker = None
    assert doc_from_dict(to_dict(doc)) == doc


def test_validate_clean():
    assert validate(make_doc()) == []


def test_validate_catches_issues():
    doc = make_doc()
    doc.replicas[0].speaker = "char_9"          # unknown id
    doc.replicas.append(Replica(5, 900, "char_0"))  # out of bounds + overlap
    doc.mentions[0] = Mention(18, 22, "nobody")  # unknown id
    doc.characters.append(Character("char_0", "Dup"))  # duplicate id
    doc.replicas[0].qtype = "weird"
    issues = "\n".join(validate(doc))
    for needle in ("char_9", "out of bounds", "overlap", "nobody",
                   "duplicate", "qtype"):
        assert needle in issues, issues
