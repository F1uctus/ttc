from pathlib import Path

import numpy as np
import pytest
import spacy

from ttc.ml.encoder import Encoder
from ttc.ml.extensions import register
from ttc.ml.packages import ModelPackage

MINI = Path(__file__).parent / "fixtures" / "mini_package"


@pytest.fixture(scope="module")
def nlp():
    return spacy.blank("ru")


def test_encode_doc_shape_and_cache(nlp):
    register()
    enc = Encoder(ModelPackage.from_dir(MINI))
    doc = nlp("Привет мир . Это длинный текст для окон .")
    emb = enc.encode_doc(doc)
    assert emb.shape == (len(doc), 8)
    assert doc._.ttc_emb is emb
    assert np.abs(emb).sum(axis=1).min() > 0


def test_windowing_covers_long_text(nlp):
    register()
    enc = Encoder(ModelPackage.from_dir(MINI))
    doc = nlp(" ".join(f"слово{i}" for i in range(100)))  # longer than the window
    emb = enc.encode_doc(doc)
    assert emb.shape == (len(doc), 8)
    assert np.abs(emb).sum(axis=1).min() > 0


def test_encoder_factory_registered(nlp):
    import ttc.ml.components  # noqa: F401  (registers factories)

    pipe = nlp.add_pipe("ttc_encoder", config={"package_dir": str(MINI)})
    doc = pipe(nlp.make_doc("Один два три"))
    assert doc._.ttc_emb is not None
    nlp.remove_pipe("ttc_encoder")
