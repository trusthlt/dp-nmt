import dp_accounting
from flax.training.common_utils import shard
from matplotlib import pyplot as plt
from flax import traverse_util
import numpy as np
import math
import warnings
import logging
import jax
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_PRIVATE_MAXIMUM_BATCH_SIZE = {
    "mbart-large-cc25": 16,
    "mt5-small": 45,
    "mt5-base": 18,
    "t5-small": 250,
    "t5-base": 60,
    "t5-large": 16
}


def numpy_collate(batch):
    batch = {key: np.array([sample[key] for sample in batch]) for key in batch[0].keys()}
    while batch['input_ids'].shape[0] % jax.local_device_count() != 0:
        for key in batch.keys():
            batch[key] = np.vstack([batch[key], batch[key][-1]])
    return shard(batch)


def sacre_bleu_postprocess_text(predictions, labels):
    predictions = [pred.strip() for pred in predictions]
    labels = [[label.strip()] for label in labels]

    return predictions, labels


def decode_postprocess_text(tokenizer, predictions, labels):
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_predictions, decoded_labels = sacre_bleu_postprocess_text(decoded_predictions, decoded_labels)
    return decoded_predictions, decoded_labels


def compute_epsilons(num_examples, batch_size, noise_multiplier, epochs, delta=1e-8):
    # delta should be < 1/num_examples
    if num_examples * delta > 1.:
        warnings.warn('Your delta might be too high.')
    orders = (
            [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            + list(range(5, 64))) + [128, 256, 512]
    steps = int(math.ceil(epochs * num_examples / batch_size))  # be clear on 'batch size'
    logger.warning("Assuming Poisson sampling for DP-SGD.")
    q = batch_size / num_examples
    if q > 1:
        warnings.warn("'n' must be larger than the batch size.")

    accountant = dp_accounting.rdp.RdpAccountant(orders)
    gaussian_event = dp_accounting.GaussianDpEvent(noise_multiplier)
    poisson_event = dp_accounting.PoissonSampledDpEvent(
            sampling_probability=q, event=gaussian_event)
    event = dp_accounting.SelfComposedDpEvent(poisson_event, steps)
    accountant.compose(event)
    epsilon = accountant.get_epsilon_and_optimal_order(delta)
    return epsilon[0]


def plot_learning_curve(train_losses, validation_loss, output_dir, file_name, combined_plot=False):
    """
    Result png figures are saved in the log directory.
    """
    if combined_plot:
        fig, ax = plt.subplots(num=1, clear=True)
        fig.suptitle('Model Learning Curve')
        steps = [str(i) for i in range(len(train_losses))]
        ax.plot(steps, train_losses, 'o-', markersize=2, color='b', label='Train')
        ax.plot(steps, validation_loss, 'o-', markersize=2, color='c', label='Validation')
        ax.set(xlabel='Step', ylabel='Loss')
        ax.legend()
        plt.savefig(os.path.join(output_dir, file_name))
        logger.info(f"Plot learning curve to: {output_dir}/{file_name}")
    else:
        for i, loss in enumerate([train_losses, validation_loss]):
            split = 'Training' if i == 0 else 'Validation'
            color = 'b' if i == 0 else 'c'
            plt.figure(num=i, clear=True)
            plt.title('Model Learning Curve')
            steps = list(range(len(loss)))
            plt.plot(steps, loss, 'o-', markersize=2, color=color, label=split)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(output_dir, split.lower() + '_' + file_name))
        logger.info(f"Plots learning curve saved to: {output_dir}")


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(params)
    # find out all LayerNorm parameters
    layer_norm_candidates = ["final_layer_norm", "layernorm", "layer_norm", "ln"]
    layer_norm_named_params = {
        layer[-2:]
        for layer_norm_name in layer_norm_candidates
        for layer in flat_params.keys()
        if layer_norm_name in "".join(layer).lower()
    }
    flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
    return traverse_util.unflatten_dict(flat_mask)
