import os
import json
import datasets
import itertools

_LANGUAGES = [
    "de",
    "en",
    "es",
    "fr",
    "pt",
    "zh",
]
_LANGUAGE_PAIRS = list(itertools.combinations(_LANGUAGES, 2)) + [tuple(reversed(ele)) for ele in list(itertools.combinations(_LANGUAGES, 2))]

_BASE_URL = os.getcwd()

_URL = {
    "train": _BASE_URL + "/data/multiatis++/train.json",
    "dev": _BASE_URL + "/data/multiatis++/dev.json",
    "test": _BASE_URL + "/data/multiatis++/test.json",
}

_DESCRIPTION = """\
    The ATIS (Air Travel Information Services) collection was developed to support the research and development of speech understanding systems [1]. The original English data includes intent and slot annotations, and was later extended to Hindi and Turkish. MultiATIS++ futher extends ATIS to 6 more languages, and hence, covers a total of 9 languages, that is, English, Spanish, German, French, Portuguese, Chinese, Japanese, Hindi and Turkish. These locales belong to a diverse set of language families- Indo-European, Sino-Tibetan, Japonic and Altaic. MultiATIS++ corpus has been outsourced to foster further research in the domain of multilingual/cross-lingual natural language understanding.
"""
_HOMEPAGE = "https://github.com/amazon-science/multiatis"

_CITATION = """\
    @inproceedings{xu-etal-2020-end,
    title = "End-to-End Slot Alignment and Recognition for Cross-Lingual NLU",
    author = "Xu, Weijia  and
      Haider, Batool  and
      Mansour, Saab",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.410",
    doi = "10.18653/v1/2020.emnlp-main.410",
    pages = "5052--5063",
    abstract = "Natural language understanding (NLU) in the context of goal-oriented dialog systems typically includes intent classification and slot labeling tasks. Existing methods to expand an NLU system to new languages use machine translation with slot label projection from source to the translated utterances, and thus are sensitive to projection errors. In this work, we propose a novel end-to-end model that learns to align and predict target slot labels jointly for cross-lingual transfer. We introduce MultiATIS++, a new multilingual NLU corpus that extends the Multilingual ATIS corpus to nine languages across four language families, and evaluate our method using the corpus. Results show that our method outperforms a simple label projection method using fast-align on most languages, and achieves competitive performance to the more complex, state-of-the-art projection method with only half of the training time. We release our MultiATIS++ corpus to the community to continue future research on cross-lingual NLU.",
}
"""


class MultiAtisPlusPlus(datasets.GeneratorBasedBuilder):
    """MultiAtis++ Corpus"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"{l1}-{l2}", version=datasets.Version("1.0.0"), description=f"MultiAtis++ Corpus {l1}-{l2}"
        )
        for l1, l2 in _LANGUAGE_PAIRS
    ]

    def _info(self):
        src, tgt = self.config.name.split("-")
        features = datasets.Features(
            {
                "translation": {
                    "intent": datasets.Value("string"),
                    src: datasets.Value("string"),
                    tgt: datasets.Value("string"),

                }
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang_pair = tuple(self.config.name.split("-"))
        data_dir = dl_manager.download_and_extract(_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "lang_pair": lang_pair,
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "lang_pair": lang_pair,
                    "filepath": data_dir["test"],
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "lang_pair": lang_pair,
                    "filepath": data_dir["dev"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, lang_pair, filepath, split):
        """Yields examples."""
        src, target = lang_pair
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, sentence in enumerate(data):
                yield idx, {
                    "translation": {
                        "intent": sentence["intent"],
                        src: sentence[src],
                        target: sentence[target],
                    }
                }
