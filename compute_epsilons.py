from datasets import load_dataset_builder
import numpy as np
import argparse
import utils


def main():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--dataset", type=str, required=True)
    arg_parser.add_argument("--lang_pair", type=str, default='de-en')
    arg_parser.add_argument("--batch_size", type=int, required=True)
    arg_parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Scale up the batch size")
    arg_parser.add_argument("--noise_multiplier", type=float, default=None)
    arg_parser.add_argument("--device_count", type=int, required=True)
    arg_parser.add_argument("--epochs", type=int, required=True)

    args = arg_parser.parse_args()

    # Set values
    total_batch_size = args.batch_size * args.device_count
    epochs = args.epochs
    ds_builder = load_dataset_builder(args.dataset, args.lang_pair)
    len_train_dataset = ds_builder.info.splits['train'].num_examples
    num_batch, remainder = divmod(len_train_dataset, total_batch_size)
    actual_compute_len_train = num_batch * total_batch_size if remainder == 0 else (num_batch + 1) * total_batch_size
    if args.noise_multiplier is None:
        noise_multipliers = np.concatenate(
            (np.arange(0.0, 1.0, 0.01),
             np.arange(1.0, 5.0, 0.2),
             np.arange(5.0, 100, 5.0),
             np.array([128, 256])
             )
        )
    else:
        noise_multipliers = [args.noise_multiplier]
    print("actual_compute_len_train:", actual_compute_len_train)
    print("devices:", args.device_count)
    print("total_batch_size:", total_batch_size)
    print("gradient_accumulation_steps:", args.gradient_accumulation_steps)
    print("accumulation_batch_size:", total_batch_size * args.gradient_accumulation_steps)
    print("epochs:", epochs)
    for noise_multiplier in noise_multipliers:
        print("input noise_multiplier:", noise_multiplier)
        _ = utils.compute_epsilons(
            actual_compute_len_train,
            total_batch_size * args.gradient_accumulation_steps,
            noise_multiplier,
            epochs
        )
        print("\n")


if __name__ == '__main__':
    main()
