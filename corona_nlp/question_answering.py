import json
import re
from pathlib import Path
from typing import IO, Dict, List, Set, Union

import spacy
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import TextToSpeechV1
from pydub import AudioSegment, playback
from spacy import displacy
from spacy.lang.en import STOP_WORDS

# template configuration for ibm-watson
config = {
    "apikey": "<YOUR_IAM_APIKEY>",
    "url": "<YOUR_API_URL>",
    "disable_ssl": False,
    "voice": "en-US_MichaelV3Voice",
}


def summarize(texts: Union[str, List[str]],
              nlargest=7,
              min_token_freq=30,
              spacy_model="en_core_web_sm",
              spacy_nlp: object = None,
              stop_words: Union[List[str], Set[str]] = None) -> str:
    """Text summarizer built on top of spaCy.

    - You want fast or quality:
        - If quality, please look at the Transformers summarizer pipeline
            I highly recommend using it as opposed to this summarizer.
        - If speed, try it.
    - This method is an alternative to using Transformers's summarizer
        pipeline - the outputs are state-of-the-art but extremely slow!
        This summarization algorithm is simple, good enough and fast!

    `texts`: An iterable list of string sequences.
    `spacy_model`: If `spacy_nlp=None`, the default nlp model will be
        used. The summarization quality improves the larger the model.
        For accuracy I recommend using `en_core_web_md`.
    `spacy_nlp`: Use an existing `spacy.lang.en.English` instance.
        The object is usually referred as `nlp`. Otherwise, a new
        instance will be loaded (which can take some time!).
    """
    nlp, doc, stop_words = (None, None, stop_words)
    if spacy_nlp is not None:
        nlp = spacy_nlp
    else:
        nlp = spacy.load(spacy_model)
    if isinstance(texts, list):
        doc = nlp(" ".join(texts))
    elif isinstance(texts, str):
        doc = nlp(texts)
    if stop_words is None:
        stop_words = STOP_WORDS

    k_words = {}  # word level frequency
    for token in doc:
        token = token.text
        if token not in stop_words:
            if token not in k_words:
                k_words[token] = 1
            else:
                k_words[token] += 1

    # normalize word frequency distributions
    for w in k_words:
        k_words[w] = k_words[w] / max(k_words.values())

    scores = {}  # sentence level scores.
    for sent in [i for i in doc.sents]:
        for word in sent:
            word = word.text.lower()
            if word in k_words:
                if len(sent.text.split()) < min_token_freq:
                    if sent not in scores:
                        scores[sent] = k_words[word]
                    else:
                        scores[sent] += k_words[word]

    # find the n[:n] largest sentences from the scores.
    sents = sorted(scores, key=scores.get, reverse=True)[:nlargest]
    summary = " ".join([i.text for i in sents])
    return summary


class WatsonTextToSpeech:

    def __init__(
        self,
        apikey: str,
        url: str,
        disable_ssl: bool = False,
        cache_dir: str = "voice_mp3",
        voice: str = "en-US_MichaelV3Voice",
        audio_format: str = "audio/wav",
        spacy_model: str = "en_core_web_md",
    ):
        """Watson text to speech utility class.

        `apikey`: watson text-to-speech iam api key.
        `url`: watson api url (the url is related to the api-key)
        `disable_ssl`: disable ssl for all requests to watson-cloud
        `cache_dir`: the directory where all the files will be saved.
        """
        self._query_index = 0
        self.text_to_speech = TextToSpeechV1(IAMAuthenticator(apikey))
        self.text_to_speech.set_service_url(url)
        if disable_ssl:
            self.text_to_speech.set_disable_ssl_verification(True)
        self.voice = voice
        self.audio_format = audio_format
        self.nlp = spacy.load(spacy_model)
        self.queries: Dict[int, Dict[str, str]] = {}
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir(parents=True)

    def list_voices(self) -> List[Dict[str, str]]:
        return self.text_to_speech.list_voices().get_result()["voices"]

    def save_meta(self, meta='meta.json', cache_dir: Path = None) -> IO:
        cache = self.cache_dir if cache_dir is None else cache_dir
        meta = cache.joinpath(meta)
        with meta.open('w') as file:
            file.write(json.dumps(self.queries, indent=4,
                                  separators=(",", ": ")))

    def load_meta(self, inplace=False, meta='meta.json',
                  cache_dir: Path = None) -> Dict[int, Dict[str, str]]:
        """Load the meta data containing the query texts and audio files."""
        cache = self.cache_dir if cache_dir is None else cache_dir
        meta = cache.joinpath(meta)
        meta_data = {}
        with meta.open("r") as file:
            meta = json.load(file)
        for index, data in meta.items():
            meta_data[int(index)] = data
        if inplace:
            self.queries = meta_data
        return meta_data

    def is_similar(self, text: str, to: str, k=.95) -> bool:
        score = self.nlp(text).similarity(self.nlp(to))
        return True if score >= k else False

    def smart_cache(self, text: str, k=.95) -> int:
        """Return the file index if the text is similar to a cached text."""
        for index, data in self.queries.items():
            if self.is_similar(data["text"], to=text, k=k):
                return index

    def play_synth(self, file: Path, suffix="wav") -> None:
        if isinstance(file, Path):
            sound = AudioSegment.from_file(file, format=suffix)
            playback.play(sound)

    def write_synth(self, file: Path, text: str) -> IO:
        with file.open("wb") as audio_file:
            respose = self.text_to_speech.synthesize(
                text=text, voice=self.voice,
                accept=self.audio_format)
            audio_file.write(respose.get_result().content)

    def synthesize(self, text: str, k=.95, suffix="wav") -> None:
        audio_file = None
        text = text.strip().lower()
        index = self.smart_cache(text, k=k)
        if isinstance(index, int):
            cache_file = self.queries[index]['file']
            audio_file = self.cache_dir.joinpath(cache_file)
        else:
            new_file = f"{self._query_index}_speech.wav"
            audio_file = self.cache_dir.joinpath(new_file)
            self.queries[self._query_index] = {
                "file": new_file,
                "text": text,
            }
            self.write_synth(audio_file, text)
            self._query_index += 1

        # play the sound file from cache or newly created
        if audio_file is not None:
            self.play_synth(audio_file, suffix=suffix)


def render(question: str, prediction: Dict[str, str], jupyter=True,
           return_html=False, style="ent", manual=True, label='ANSWER'):
    """Spacy displaCy visualization util for the question answering model."""
    options = {"compact": True, "bg": "#ed7118", "color": '#000000'}
    display_data = {}
    start, end = 0, 0
    match = re.search(prediction["answer"], prediction["context"])
    if match and match.span() is not None:
        start, end = match.span()

    display_data["ents"] = [{'start': start, 'end': end, 'label': label}]
    options['ents'] = [label]
    options['colors'] = {label: "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    if len(prediction['context']) > 1:
        display_data['text'] = prediction['context']

    display_data['title'] = f'Q : {question}\n'
    if return_html:
        return displacy.render([display_data], style=style,
                               jupyter=False, options=options, manual=manual)
    else:
        displacy.render([display_data], style=style,
                        page=False, minify=True,
                        jupyter=jupyter, options=options, manual=manual)
