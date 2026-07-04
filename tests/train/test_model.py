import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.train

TINY = "hf-internal-testing/tiny-random-BertModel"


@pytest.fixture(scope="module")
def model():
    from train.model import AttributionModel

    m = AttributionModel(TINY, dim=None, hidden=16)
    m.eval()
    return m


def test_shapes_and_losses(model):
    ids = torch.randint(5, 90, (2, 12))
    mask = torch.ones_like(ids)
    hidden = model.encode(ids, mask)
    assert hidden.shape[:2] == (2, 12) and hidden.shape[2] == model.dim

    bio = torch.zeros(2, 12, dtype=torch.long)
    bio[0, 3] = 1
    assert model.cue_loss(hidden, bio).ndim == 0

    pooled = model.pool(hidden[0], [(0, 3), (4, 6)])
    assert pooled.shape == (2, model.dim)

    loss = model.ranker_loss(
        hidden[0],
        (0, 3),
        [(4, 6), (7, 9)],
        torch.tensor([[0.1, 1.0], [0.3, 0.0]]),
        gold=0,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
