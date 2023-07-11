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

[russian/*/actor_classifier.py](ttc/language/russian/pipelines/actor_classifier.py):
![](http://www.plantuml.com/plantuml/svg/ZLLHYzf047xVNp5SNxIWnTAZ89vSMoeSs4FmqHD9udWknjsmMTlQMkWFw6DV--VkbtGdkycOfENqGMTsltpVV3DClCrjPEnXdugjN2FWFhD7M6EU69bPgPN8Tr9baOdsa4g5zfWXQ6EtMfpTE0wBy5q0JBIoS_aDOV0sPFVH7aUZJrhfyHW-w8EHQCGF8QoqAKAIHdakDyTLb5XjSY4MBvVGwurXgdKcFE_LdKL3GNm4wyfACR-2AFSG-e8XzFjzM6G6FozzQ31BPHA5JE5NMuIYf2He7zLwHQRW1WA-NtIZE2mavgxuAI5UO1nVNx_nSgw-wgPYdB7cy0PclfY2PEQYUROJvIFqV7KOVjbpoVSWHl4O03eh687i3RfmOfVGWq5XQFGg7Z-e0jifGFGvqUbvInRJoJAWe2C9nB2ddx-vNHufVDce3gc5SdPL2ne-F6e8Jtz-_QSNJdN6LPDemV_E3VvH-UT5e27A-iLvqcwIyqZqYw6hl1YTL8-50bloMXkBPXu0bHDcRG2USPDZP99jfnJiDYoLDA9EojOOK4lRzbYOJBf9EPkerWxh9rK9lyUDDlYHhjXUMvvX5viakRR5G24gGZR8ruAJIyr9bQEXruwBwOHam_gU-RmhKPNLPXqju6LcA_ccNoOI3MPspI3HoDlUouLztMqjAuMrZJx18aRx1L7nq0C0Q9kVt-H9bAPnbEpQk_mFCuo-h6U-jbR7IXPH1jmelgkdhJOrDE-rmk17fdLUEWdrKiXC_C9sVopY5jNwi4__0W00)
