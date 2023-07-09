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

[russian/*/speaker_classifier.py](ttc/language/russian/pipelines/speaker_classifier.py) `[classify_speakers]`:
[![](https://mermaid.ink/img/pako:eNqNVdFumzAU_RXLT4kUaEhKk7A0UqeuUqVOipQ9DaLKgUuCCgaB0zbLIu0j9oX7kl1sICSFtjwgrn187znH5npP3dgDalE_jF_cDUsF-XHrcILPQmDU8QIWxustEB4LAlEidl2iacRnls80dwPuE4Yz8hDHie3QewEpE6AS7HVdJ3aSwvOSyM8UkjBwWRFxeBXy8-DQpcPVmjyPTLhgETwEHDr5eh249xhiRK5JkUTPcnpysFsuLtc0EZxjmkXyhBwVNn_mR0bWNUaSqpqWlCRGratEFtPynYldCGQOfBsKIlCOxsJgza0QfPGljZIIIsgUJbmw49BpvgezKZafOXy6mnUeyZQEmtElv6cXqxmpfCQaMaRlCiYh-XMGWxL74W4pMUGptcD8-_O3tg8F7CKvPL2QLBxa2VkIa3LzJstQqAK8a2kIvDPHE0MG58bWU9TcPSndYNqNj0dMHs2Arzv10rl8VGxbqCsrAEspuuJUUqgknmRrF1oi3pGKlTNgqbvplLW7ee1G0bWCp7Jb6BzVfwU_TuEb987FH7nkFuAuowkgYWqX61wq9fVs7eLV_Kekq4rtwqtSp7KbeZyrvucZoDExtz8SHpRIpZ3IYXUSZNM541YnURVp96OZR5slFZd2V-o1T41RDeaOheGKIYfmFtNK_GhgmcFu6DWBNui-bSCnPUZN2ysQLwD8TXv5dGs5MbyS1ep0jfbHRhfk2m2u1Stdpj0aQRqxwMMbcJ-DHSo2EIFDLfz0wGfYhhzaU1PVHelQYpE9jrjb9Blk5NBMQCJ_YIwPDj9gcrYV8WLHXYrqwgx6dJt4eD3eBmydsqgaTRj_GccYi3SLIXiBiNPv6l6W17OEUGtPX6mlXfZNvW8MJubV6GpkTobjHt3h8GSgj4f98dDEl2H2jdGhR3_JrIY-MozhcDIx--aVYQ6Gl4f_K3N3jA?type=png)](https://mermaid.live/edit#pako:eNqNVdFumzAU_RXLT4kUaEhKk7A0UqeuUqVOipQ9DaLKgUuCCgaB0zbLIu0j9oX7kl1sICSFtjwgrn187znH5npP3dgDalE_jF_cDUsF-XHrcILPQmDU8QIWxustEB4LAlEidl2iacRnls80dwPuE4Yz8hDHie3QewEpE6AS7HVdJ3aSwvOSyM8UkjBwWRFxeBXy8-DQpcPVmjyPTLhgETwEHDr5eh249xhiRK5JkUTPcnpysFsuLtc0EZxjmkXyhBwVNn_mR0bWNUaSqpqWlCRGratEFtPynYldCGQOfBsKIlCOxsJgza0QfPGljZIIIsgUJbmw49BpvgezKZafOXy6mnUeyZQEmtElv6cXqxmpfCQaMaRlCiYh-XMGWxL74W4pMUGptcD8-_O3tg8F7CKvPL2QLBxa2VkIa3LzJstQqAK8a2kIvDPHE0MG58bWU9TcPSndYNqNj0dMHs2Arzv10rl8VGxbqCsrAEspuuJUUqgknmRrF1oi3pGKlTNgqbvplLW7ee1G0bWCp7Jb6BzVfwU_TuEb987FH7nkFuAuowkgYWqX61wq9fVs7eLV_Kekq4rtwqtSp7KbeZyrvucZoDExtz8SHpRIpZ3IYXUSZNM541YnURVp96OZR5slFZd2V-o1T41RDeaOheGKIYfmFtNK_GhgmcFu6DWBNui-bSCnPUZN2ysQLwD8TXv5dGs5MbyS1ep0jfbHRhfk2m2u1Stdpj0aQRqxwMMbcJ-DHSo2EIFDLfz0wGfYhhzaU1PVHelQYpE9jrjb9Blk5NBMQCJ_YIwPDj9gcrYV8WLHXYrqwgx6dJt4eD3eBmydsqgaTRj_GccYi3SLIXiBiNPv6l6W17OEUGtPX6mlXfZNvW8MJubV6GpkTobjHt3h8GSgj4f98dDEl2H2jdGhR3_JrIY-MozhcDIx--aVYQ6Gl4f_K3N3jA)
