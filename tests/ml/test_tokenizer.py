from ttc.ml.tokenizer import HashTokenizer


def test_hash_tokenizer_ids_and_offsets():
    tok = HashTokenizer(vocab_size=64)
    enc = tok.encode("Привет , мир")
    assert len(enc.ids) == len(enc.offsets) == 3
    assert all(1 <= i < 64 for i in enc.ids)
    assert enc.offsets[0] == (0, 6)
    assert enc.ids == tok.encode("Привет , мир").ids
