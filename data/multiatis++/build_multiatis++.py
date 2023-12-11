import pandas as pd
import json
import os

path = os.path.join("original", "data")  # MultiATIS++ dataset path
for partition in ["train", "dev", "test"]:
    intent = []
    data = {}
    for path, directories, files in os.walk(path):
        for filename in files:
            if partition in filename and "TR" not in filename and "HI" not in filename and "JA" not in filename:
                df = pd.read_csv(os.path.join(path, filename), sep="\t")
                if len(intent) == 0:
                    intent = df["intent"].to_list()
                print(filename.split("/")[-1])
                data[filename.split("/")[-1]] = df["utterance"].to_list()
    json_data = []
    for idx, inte in enumerate(intent):
        data_to_json = {}
        data_to_json["intent"] = inte
        for key, value in data.items():
            key = key.split(".")[0]
            key = key.split("_")[-1].lower()
            data_to_json[key] = value[idx]
        json_data.append(data_to_json)
    with open(f"{partition}.json", "w") as f:
        json.dump(json_data, f, indent=4)

