from settings import Settings, parse_arguments
from experiments import *
import numpy as np
import logging
import jax

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    # get the settings from the command line
    settings = Settings(parse_arguments()).args
    np.random.seed(settings.seed)

    logger.info(f"Dataset: {settings.dataset}")
    logger.info(f"Model: {settings.model}")
    logger.info(f"Seed: {settings.seed}")
    logger.info(f"Count device: {jax.local_device_count()}")
    if not settings.generate:
        if settings.freeze:
            logger.info("Training only on Cross Attention layers")
        else:
            logger.info("Training on all layers")
        logger.info(f"Lang Pair: {settings.lang_pair}")

        if settings.num_examples is not None:
            logger.info(f"Train on {settings.num_examples} training examples")
        else:
            logger.info(f"Train on full training data")
    
        logger.info(f"Evaluate on test Set: {settings.test}")

        logger.info(f"Epochs: {settings.epochs}")
        logger.info(f"Batch Size: {settings.batch_size}")
        logger.info(f"Gradient Accumulation Steps: {settings.gradient_accumulation_steps}")
        logger.info(
            f"Gradient Accumulation Batch Size: {settings.gradient_accumulation_steps * settings.batch_size}")
        logger.info(f"Optimizer: {settings.optimizer}")
        logger.info(f"Learning Rate: {settings.learning_rate}")
        logger.info(f"Max sequence length: {settings.max_seq_len}")

        logger.info(f"Early Stopping: {settings.early_stopping}")
        if settings.early_stopping:
            logger.info(f"Patience: {settings.patience}")
            logger.info(f"Minimum delta between updates: {settings.early_stop_min_delta}")

        if settings.private:
            logger.info(f"Private training")
            logger.info(f"L2 Norm Clip: {settings.l2_norm_clip}")
            logger.info(f"Noise Multiplier: {settings.noise_multiplier}")
        else:
            logger.info(f"Normal training inf epsilon")

        logger.info(f"Poisson sampling: {settings.poisson_sampling}")

    # Load
    logger.info(f"Loading experiment")

    if "mbart" in settings.model:
        experiment = MBartExperiment(settings)
    elif "mt5" in settings.model:
        experiment = MT5Experiment(settings)
    elif "t5" in settings.model:
        experiment = T5Experiment(settings)
    else:
        raise ValueError("Model is not supported")

    experiment()
    if settings.generate:
        experiment.run_generate()
    else:
        experiment.run_experiment()


if __name__ == '__main__':
    main()
