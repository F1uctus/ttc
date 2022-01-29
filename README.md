## Text-To-Conversation toolkit (TTC)

### Usage

Install [Poetry](https://python-poetry.org) to your Python distribution.

Spawn a new virtual environment for the project:

```shell
poetry shell
```

Install project dependencies:

```shell
poetry install
```

Test output on a text file:

```shell
- poetry run print_play path_to_the_text_file text_language
```

**Notes**

- Text must be encoded in UTF-8;
- It is usually better to test on some middle-sized text (e.g a book chapter);
- Supported `text_language`s are:
    + ru (russian)
