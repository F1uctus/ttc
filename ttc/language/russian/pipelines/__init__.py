from spacy import Language

import ttc.language.russian.pipelines.line_numerator as line_number_marker
import ttc.language.russian.pipelines.mispredictions_corrector as mis_corrector
import ttc.language.russian.pipelines.sentencizer as sentencizer


def register_for(lang: Language) -> None:
    if not lang.has_pipe(sentencizer.NAME):
        lang.add_pipe(sentencizer.NAME, after="morphologizer")

    if not lang.has_pipe(line_number_marker.NAME):
        lang.add_pipe(line_number_marker.NAME, after=sentencizer.NAME)

    if not lang.has_pipe(mis_corrector.NAME):
        lang.add_pipe(mis_corrector.NAME, after="parser")
