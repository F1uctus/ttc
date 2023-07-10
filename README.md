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
![](http://www.plantuml.com/plantuml/svg/ZLLHYzf047wVNp5SNxIWnTAZ89vSMoeSs4FmqHFPw7Wk9fkmMTlQMkWFw6DV--VkbtGtsUoOfENqGMTsltpVV3DClCsDqEQGnB2J6q9ACdF43ULhBJCZKmNvNgfCQ95WB1MPOqQOQhDBuUp6SXZ2xu0uIPMPos-4WxS-knS9ZKQVKfMEn_WXFMX96du066bYmdKiybnkZwiy8x4ddGCiNYwnrnlZD4qpA9clxWnf5qH7DBQmcF8h57a9SLyun7w_ZujCq-Tb6MhAOhaMFbFqrHGXL9GEsYUrMJbRU8CUto_wKHmM4dEHqDGXNb0KNL-_AULiVU4cCCvOStY3ixAOWcHcetcsX-eHztmuzB_1S-1xW54qHiJE2aSeUuCkdjWbzd1G61gzYiOFgi1s2f3xdDpfUKiMqydIex0Z7OXXJpz_SxjA99Hbaph41glfXHRKVNZK49_-_FfFBwpgZAjseWN_EplwHrK-3vx6KTOlpfDsavn7IxyOsigBqKdrM2YmfL9hQqZFFI0uOTOwu3atEGcztdKgmMx3KaK3DLAsne1wshR7miwadPIrIMfZiUMaWl1xsgQQFjehjjUMPzZb9frSswAd02OC1laQEBdKd5GvQVVYiR4xSR9-VSzytbMeOBLPHnDueTaWt_JBHD1WPjSCWaRUzbujxEjkUrbjMDle4ouYCb-850_T0z1jy_ERF15n78drlhNB_pB3wCjwvcjhTgnYWIfmtCsxUjfkKrtpFbNa_SYupkjEVBqkP2P-OTi_5hWbjJaayLy0)
