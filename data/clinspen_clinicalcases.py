import os
import json
import datasets

_BASE_URL = os.getcwd()

_DESCRIPTION = ""

_HOMEPAGE = "https://temu.bsc.es/clinspen/clinical-cases/"

_LICENSE = ""

_CITATION = """\
    @inproceedings{neves2022findings,
      title={Findings of the WMT 2022 biomedical translation shared task: Monolingual clinical case reports},
      author={Neves, Mariana and Yepes, Antonio Jimeno and Siu, Amy and Roller, Roland and Thomas, Philippe and Navarro, Maika Vicente and Yeganova, Lana and Wiemann, Dina and Di Nunzio, Giorgio Maria and Vezzani, Federica and others},
      booktitle={WMT22-Seventh Conference on Machine Translation},
      pages={694--723},
      year={2022}
}

"""

_URL = {
    "train": _BASE_URL + "/data/clinspen_clinicalcases/train.json",
    "dev": _BASE_URL + "/data/clinspen_clinicalcases/dev.json",
    "test": _BASE_URL + "/data/clinspen_clinicalcases/test.json",
}


class ClinSpEn(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    
    def _info(self):
        features = datasets.Features(
            {
                "translation": {
                    "file_name": datasets.Value("string"),
                    "en": datasets.Value("string"),
                    "es": datasets.Value("string"),
                }
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"], 
                    "split": "test"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["dev"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for idx, sentence in enumerate(data):
                yield idx, {
                    "translation": {
                        "file_name": sentence["file_name"],
                        "en": sentence["en"],
                        "es": sentence["es"],
                    }
                }
