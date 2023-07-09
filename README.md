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
![](http://www.plantuml.com/plantuml/svg/ZLLHYzf047wVNp5SNxo25akFWgXSMoeSs4FmqHFPw7Wk9fkmMTlQMkWFw6DV--VkbtGtsUoOfENqGMTsltpVV3DClCsDqEQGnB2J6q9ACdF43ULhBJCZKmNvNgfCQ95WB1MPOqQOQhDBuUp6SXZ2xu0uIPMPos-4_RS-knS93OUVKfMEn_WXFMX96du066bYmdKiybnkZwiy8x4ddGCiNYwnsntZD4qpA9clxWnf5qH7DBQmcF8h57a9SLyum5wlXujCq-Tb6MhAOhaMFbFqrHGXL9GEsYErMJbRE4AFxnNzA0uBYRb8QEeGBoYARcxUb7Asld0J66UiEHZXh2oc89dPQ9xjeNh4FHyE_M_m7FWUO1YD4R6pmY7ARlyQJ-mIkzWl30rVHUC7LM0x1SXzJkxql2MBwMJfKTYH3iImfv-_kLqb4igoIHrYWzNqGWlgFZngW4z_Vlsd5rPrnbKxqOB_dHtzeogV1q_ZAEiNvqcxISwZfLyCREN5wAHwB1JOKgcrDQJd7X2SC6iTy9oRd8HUxpgLu7N3KaK3DLAsne1wshR7miwadPIrIMfZiUMaWl1xsgQQFjehjjUMPzZb9frSswAd02PaW-4QEBdKd5GvQVVYiR4xSR9-VSzytbMeOBLPHnDueTaWt_JBHD1WPjSCWaRUzbujxElhzhBQiBRH9rn4PBuGAHww1w3RvkStUIBYE1BhVMkN_sM6qPVrpDVMx5Z50bNWk9jtzRJMfgRc2EzJHTwEZEkqxiINwq8culUr_L-1MrARGnB_1G00)
