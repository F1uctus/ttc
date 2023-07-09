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
![](http://www.plantuml.com/plantuml/svg/ZLFDYbe_5D_dAGwpSlwWV-oc82hJbWe3TX1SEY9HZsCm9fSajhKVq0Veijk-t3n9Sq-I6--zPUf6St9--NsSoxcpZXjtEah6zcABWCV2dM6BTcD4uOHMp1w4AhZXHv12eJiN2DgulMON7K_Y4BuoW8bMRYw-8FHVn-g17t4u_A2L7e_XdJuPWOPzPym99n4sabihTkULBP0Vq5Z65lyjeTiTmrJhWWNcctk7fZoip-2yCOds1hWDaF871j3hzM1H6Fou34U3XHGR7Yk5dnqTMQ8ieJrKsrKP2qOGyJqQ25sInDpRdvO8qY5ZwQgYccMnmGXcmGMzdldJl3Ymz8JUfyludyCclaSsN8y1eBE28OXk_nQ-3VzVZtD4w4AVqZmgmdO8101Du18Mqyaoeg0Z25J1dd_yodCAI1Piepf91-cpiOR_q1t0y--VByw0R1BX6sl5ameFMNnFNBN6PT6UIXdAfrWCl5NwI48zBIb7aevQEuTc7W4iizKRBzhtCXQvsUmxIUYs5Bg8Kl7NI7Ea3AYNxLYLC8NpdyJ7GxNrsF19Ak6tkDC6trEhBPlRruSZcLQqIC5gX3bOCf2BAqC9PT7RrIha0hekjUDsNeIyJz2MYIRWogILTZDi9c9DvXmvssdKxU1f7L-rThQzhm6mMAFxXAZggqgx-iyzdN0frtnpo9lwbn0PTTsiWf_Mf18BImVYRlJgvSLusfeZUD0AOmmXQxHq4-tICHFTeTgUZl8F)
