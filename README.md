## Text-To-Conversation toolkit (TTC)

This NLP library can help you with:

- Extraction of characters' replicas from literary texts;
- Identification of the actors owning these replicas.

### Demo (CLI)

![](assets/demo-cli.png)

### Progress

We aim to achieve the following goals:

- Better accuracy on the actor classification task
  (it is near 80% or worse for now);
- Support for more languages
  (only Russian is supported at the moment).

### Installation

Install with pip is just a usual `pip install .` from the project dir.

### Usage

**As a library**

You can find an example of using the library in the [`cli.py`](ttc/cli.py) file.

**As a tool for the CLI**

Test output on a text file:

```console
ttc print-play path-to-the-text-file text-language
```

**Notes**

- Text must be encoded in UTF-8;
- Text must be sanitized (see https://github.com/F1uctus/ttc/issues/23);
- It is usually better to test on some middle-sized text (e.g a book chapter);
- Supported `text-language`s are:
    + ru (russian)

### Development

Please install [Poetry](https://python-poetry.org).

Spawn a new virtual environment for the project:

```console
poetry shell
```

Install project dependencies:

```console
poetry install [--with dev,large_models_ru]
```

Contributions are very welcome!

#### Implementation notes

[russian/*/speaker_classifier.py](ttc/language/russian/pipelines/speaker_classifier.py):
![](http://www.plantuml.com/plantuml/svg/ZPFRYXfF4CRl-ocA_T-uVz1Wo18GbKr2X0Mp25ww8ZDQherZzz3T9Z47o0FaChTvkNsIz5Jtb7F8mHljmpVVLxzgwjQwnBZZ8HCxkK70G-vEi46xDZ9tKYjXzrBbYKaEa4c5xfGZQEDsMbHkl8T3-2m09bgvkVo4q7yPT_V90OV3TrhfyHZUwAEHQCHN8Pnq6S8wIwoLszFAvfZiqLWX5lyleTiTmrJhN0Jdgpk7fbYa9t2-i0pj5IGsI8eV642lruD5Rl3zCYmDvfbS9t6dyAEZfIWj2saFrMPLOC48ehv7_Q13YfZxygU5uZzCqvkR5nHNrCStx3WxPmv6C0l5aCdCh-RvNh2P_vuFupypiz-36APZ0EYiO0Yosx-6ByDd_eH9eGjzWdfGfEsG202QmYKiffDbL457aeXbJzz-SBkYKeIoKHqp1_nwPGiQ744rWAUVt__O2owNZ5zPAnzL-534xychTb7TjBjJm7cQnS2NJ6ycrUEIU3WwP6qTcdaKCCBPjI7u6uQFivYOzQvJ1buNWHMaqaRKJHdjBAXlstWSJ93-YJmcgetNXYVYuDUurGRVqXNZjUDPTZaanRKGeuLWmmeWNmlUN0mdbKLVhbQ13ufZeep3Mu6zAwNCsooc9RoCc6jeGf-DHDEPqSyYqTYt6r2Ew_FMJBVdEGWMAReFYAe-oZJB_vpZIP9bQRBUjoVvDp2ilh1deQvMnqeNKIx8kt5NtxRQLCqSmRrM675aLhEbarWkBTde5jNcUCX-0G00)
