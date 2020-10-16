import random
import re
from typing import (IO, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple,
                    Union)

import colour
from colour import C_HEX, COLOR_NAME_TO_RGB, Color, color_scale
from spacy import displacy


def render_output(
    output=None,
    answer: Optional[str] = None,
    context: Optional[str] = None,
    question: Optional[str] = None,
    spans: Optional[List[Tuple[int, int]]] = None,
    style: str = "ent",
    manual: bool = True,
    jupyter: bool = True,
    return_html: bool = False,
    answer_label: str = 'A',
    title_label: str = 'Q',
    gradient: Sequence[str] = ["90deg", "#aa9cfc", "#fc9ce7"],
    options: Optional[Dict[str, Any]] = None,
):
    """A displaCy visualizer for QA outputs."""

    def hex_color_pairs(n: int) -> List[Tuple[str, str]]:
        c_name = list(COLOR_NAME_TO_RGB.keys())
        sample = random.sample(c_name, k=n * 2)
        hexseq = [Color(c).get_hex_l() for c in sample]
        return list(zip(hexseq[n:], hexseq[:n]))

    def flatscore(k, size, ndigits=2):
        return round((k * 10**8) * 100 - size, ndigits=ndigits)

    DOC, OPTIONS = None, None
    H1 = f'\n{title_label}:' + '\t{text}\n'

    if output is not None and all(
            hasattr(output, attr) for attr in ('q', 'c', 'ids', 'dist')
    ):
        title = H1.format(text=output.q)
        docs = dict(text=output.c, title=title, ents=[])
        size = output.size()
        for pred in output:
            score = flatscore(pred.score, size, ndigits=2)
            xlabel = f'{answer_label} ({score} %)'
            start, end = pred.start, pred.end
            docs['ents'].append(dict(start=start, end=start, label=xlabel))

        labelmap = {}
        hexpairs = hex_color_pairs(output.size())
        ent_keys = [x['label'] for x in docs['ents']]

        for ent_key, (hex1, hex2) in zip(ent_keys, hexpairs):
            gradient = f'linear-gradient({90}deg, {hex1}, {hex2})'
            if ent_key not in labelmap:
                labelmap[ent_key] = gradient
        colorcfg = dict(ents=ent_keys, colors=labelmap)
        DOC, OPTIONS = [docs], colorcfg

    elif all((answer, context)):
        doc = dict(text=context, ents=[])
        if question is not None and isinstance(question, str):
            title = H1.format(text=question)
            doc.update({'title': title})

        start, end = spans if spans is not None else (0, 0)
        if spans is None:
            match = re.search(answer, context)
            if match and match.span() is not None:
                start, end = match.span()
                doc['ents'] = [dict(start=start, end=end, label=answer_label)]

        colorcfg = {'ents': {}, 'colors': {}}
        gradient = ", ".format(gradient)
        colors = "linear-gradient({})".format(gradient)
        colorcfg.update({'ents': [answer_label],
                         'colors': {answer_label: colors}})
        DOC, OPTIONS = [doc], colorcfg

    if options is None:
        if style == "dep":
            options = dict(compact=True, bg="#ed7118", color="#000000")

    if return_html:
        return displacy.render(DOC, style=style, jupyter=False,
                               options=OPTIONS, manual=manual)
    else:
        displacy.render(DOC, style=style, page=True, minify=False,
                        jupyter=jupyter, options=OPTIONS, manual=manual)
