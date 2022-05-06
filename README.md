## Text-To-Conversation toolkit (TTC)

### Usage

Install [Poetry](https://python-poetry.org) to your Python distribution.

Spawn a new virtual environment for the project:

```console
$ poetry shell
```

Install project dependencies:

```console
$ poetry install
```

Test output on a text file:

```console
$ poetry run print-play path-to-the-text-file text-language
```

**Notes**

- Text must be encoded in UTF-8;
- Text must be sanitized (see https://github.com/F1uctus/ttc/issues/23);
- It is usually better to test on some middle-sized text (e.g a book chapter);
- Supported `text-language`s are:
    + ru (russian)
