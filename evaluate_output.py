from evaluate import load, combine
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data', type=str, required=True)
    arg_parser.add_argument('--lang', type=str, default='en')
    args = arg_parser.parse_args()
    data_name = args.data.split('.')[0]
    with open(args.data, 'r') as f:
        data = json.load(f)

    bertscore = load("bertscore")

    references = data['references']
    predictions = data['predictions']
    results = bertscore.compute(predictions=predictions, references=references, lang=args.lang, verbose=True)
    data['bertscore_f1'] = sum(results['f1']) / len(results['f1'])

    logger.info(f"BERTScore F1: {sum(results['f1']) / len(results['f1'])}")

    with open(f"{data_name}_bertscore.json", 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()
