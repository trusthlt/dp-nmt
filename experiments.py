from jax import lax
from transformers import AutoTokenizer, MBartTokenizer, FlaxAutoModelForSeq2SeqLM
from flax.training import train_state, early_stopping
from opacus.data_loader import DPDataLoader
from dataloader import CustomDPDataLoader
from flax.core.frozen_dict import freeze
from torch.utils.data import DataLoader
from flax.jax_utils import unreplicate
from datetime import datetime
from preprocessing import *
import jax.numpy as jnp
from typing import Any
from tqdm import tqdm
from utils import *
import evaluate
import logging
import optax
import flax
import glob
import json
import jax
import os
import time
import pdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainState(train_state.TrainState):
    accum_grads: flax.core.FrozenDict[str, Any]


class Experiment:
    def __init__(self, settings):
        # Core
        self.data_to_write: Any = vars(settings)
        self.seed = settings.seed
        self.experiment_path = "/".join(settings.model.split("/")[:-1])
        self.model = self._init_model(settings)
        self.epochs = settings.epochs
        self.metrics = evaluate.load("sacrebleu")
        self.save_path = settings.save_path
        self.device_count = jax.local_device_count()
        self.physical_batch_size = settings.batch_size
        self.batch_size_eval = settings.batch_size
        self.lot_size = settings.lot_size
        self.poisson_sampling = settings.poisson_sampling
        if self.poisson_sampling:
            self.total_batch_size = settings.lot_size
        else:
            self.total_batch_size = settings.batch_size * self.device_count
        self.test = settings.test

        # Early stopping
        self.early_stopping = settings.early_stopping
        self.patience = settings.patience
        self.early_stop_min_delta = settings.early_stop_min_delta

        # Generation
        self.num_beams = settings.num_beams
        self.max_seq_len = settings.max_seq_len
        self.generate = settings.generate

        # Privacy
        self.private = settings.private
        self.noise_multiplier = settings.noise_multiplier
        self.l2_norm_clip = settings.l2_norm_clip

        # Training log
        self.training_loss_step = []
        self.validation_loss_step = []
        self.training_loss_epoch = []
        self.validation_loss_epoch = []
        self.predictions = []
        self.references = []

        # Memory allocation
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = settings.preallocate_memory

        # Saving folder
        current_time = datetime.now()

        # If os.makedirs(self.write_dir) fails, then sleep for 1 second and try again
        if not settings.generate:
            back_off = 10
            for _ in range(back_off):
                try:
                    current_time = datetime.now()
                    self.save_time = current_time.strftime("%Y_%m_%d-%H_%M_%S")
                    self.write_dir = self.save_path + self.save_time

                    if not os.path.exists(self.write_dir):
                        os.makedirs(self.write_dir)
                        self._dump_config(settings)
                        break
                    else:
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Error: {e}")

        # To be written if condition met
        self.result = {}
        self.epsilon = None
        self.early_stop_epoch = None
        self.start_time = current_time.strftime("%Y_%m_%d-%H_%M_%S")

        # To be overwritten in subclass
        self.model_name = ""
        self.tokenizer = None
        self.decoder_start_token_id = None
        self.preprocessor = Preprocessor(
            model=self.model,
            tokenizer=self.tokenizer,
            lang_pair=settings.lang_pair,
            max_seq_len=settings.max_seq_len
        )

        # Continue training
        self.resume_from_epoch = settings.resume_from_epoch
        if self.resume_from_epoch != 0:
            self._load_checkpoint_data(settings)
        else:
            logger.info(f"Start new training")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _dump_config(self, settings):
        file_path = "config.json"

        data_to_write: Any = vars(settings)
        data_to_write['experiment_name'] = self.write_dir.split("/")[-1]

        if not settings.private:
            data_to_write.pop("l2_norm_clip")
            data_to_write.pop("noise_multiplier")

        if settings.optimizer != "AdamW":
            data_to_write.pop("weight_decay")

        with open(f"{self.write_dir}/{file_path}", "w") as file:
            json.dump(self.data_to_write, file, indent=4, separators=(',', ': '))

        logger.info(f"Write config file to: {self.write_dir}/{file_path}")

    def _init_model(self, settings):
        if settings.resume_from_epoch != 0:
            model = FlaxAutoModelForSeq2SeqLM.from_pretrained(settings.model, seed=self.seed)
        else:
            model = FlaxAutoModelForSeq2SeqLM.from_pretrained(settings.model, seed=self.seed, from_pt=True)
        return model

    def _setup_dataloader(self, settings):
        if settings.test:
            test_dataset = self.preprocessor.process_data(
                data_name=settings.dataset,
                num_examples=settings.num_examples,
                test_only=True
            )
            self.test_data_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size_eval,
                collate_fn=numpy_collate
            )
            _, self.last_eval_batch_length = divmod(len(test_dataset), self.batch_size_eval)
        else:
            if settings.generate:
                eval_dataset = self.preprocessor.process_data(
                    data_name=settings.dataset,
                    num_examples=settings.num_examples,
                    eval_only=True
                )
            else:
                train_dataset, eval_dataset = self.preprocessor.process_data(
                    data_name=settings.dataset,
                    num_examples=settings.num_examples
                )

                self.train_data_loader = DataLoader(
                    train_dataset,
                    batch_size=self.total_batch_size,
                    collate_fn=numpy_collate,
                    shuffle=True
                )

                if settings.poisson_sampling and self.private:
                    if settings.custom_dp_dataloader:
                        self.train_data_loader = CustomDPDataLoader.from_data_loader(
                            self.train_data_loader,
                            model_name=self.model_name,
                            lot_size=self.lot_size,
                            physical_batch_size=self.physical_batch_size
                        )
                    else:
                        self.train_data_loader = DPDataLoader.from_data_loader(self.train_data_loader)

                self.num_train_steps = len(train_dataset) // self.total_batch_size * self.epochs

            _, self.last_eval_batch_length = divmod(len(eval_dataset), self.batch_size_eval)

            self.eval_data_loader = DataLoader(
                eval_dataset,
                batch_size=self.batch_size_eval,
                collate_fn=numpy_collate
            )

    def _setup_optimizer(self, settings):
        # Create learning rate schedule
        warmup_fn = optax.linear_schedule(
            init_value=settings.learning_rate, end_value=0,
            transition_steps=settings.warmup_steps
        )
        decay_fn = optax.linear_schedule(
            init_value=settings.learning_rate,
            end_value=0,
            transition_steps=self.num_train_steps - settings.warmup_steps,
        )
        self.linear_decay_lr_schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, decay_fn], boundaries=[settings.warmup_steps]
        )

        if settings.optimizer == "SGD":
            self.optimizer = optax.sgd(self.linear_decay_lr_schedule_fn)
        elif settings.optimizer == "Adam":
            self.optimizer = optax.adam(self.linear_decay_lr_schedule_fn)
        elif settings.optimizer == "AdamW":
            self.optimizer = optax.adamw(
                self.linear_decay_lr_schedule_fn,
                weight_decay=settings.weight_decay,
                mask=decay_mask_fn
            )
        else:
            raise ValueError(f"No optimizer with the name {settings.optimizer}. See help")

        self.gradient_accumulation_steps = settings.gradient_accumulation_steps
        self.optimizer_name = settings.optimizer

    def _load_checkpoint_data(self, settings):
        logger.info(f"Continue training from epoch {settings.resume_from_epoch}")

        # checkpoints/date-time/model-name -> checkpoints/date-time/
        path = "/".join(settings.model.split("/")[:-1])
        train_loss = sorted(glob.glob(f'{path}/train_loss*'))
        prev_train_loss = f"{path}/train_loss_epoch_{settings.resume_from_epoch}.npy" \
            if f"{path}/train_loss_final_step.npy" not in train_loss else f"{path}/train_loss_final_step.npy"
        with open(prev_train_loss, 'rb') as f:
            self.training_loss_step = jnp.load(f).tolist()

        results = sorted(glob.glob(f'{path}/result*'))
        prev_result = f"{path}/result_epoch_{settings.resume_from_epoch}.json" \
            if f"{path}/result_final_step.json" not in results else f"{path}/result_final_step.json"
        with open(prev_result, "r") as f:
            loaded_data = json.load(f)
            self.training_loss_epoch = loaded_data["train_loss"]
            self.validation_loss_epoch = loaded_data["eval_loss"]

        if settings.resume_from_epoch != len(self.training_loss_epoch):
            raise ValueError(f"Resume training should from epoch {len(self.validation_loss_epoch)} "
                             f"instead of {settings.resume_from_epoch}")

        logger.info(f"Loaded previous checkpoint from {settings.model}")
        logger.info(f"Loaded training/eval loss per epoch from {prev_result}")
        logger.info(f"Loaded training loss per step from {prev_train_loss}")

    def train(self, parallel_train_step, state, dropout_rngs, parallel_optimizer_step=None):
        train_metrics = None
        for idx, batch in enumerate(tqdm(self.train_data_loader)):
            if "skip_batch" in batch.keys():
                logger.info("Skip empty batch")
                continue

            state, train_metrics, dropout_rngs = parallel_train_step(state, batch, dropout_rngs)
            self.training_loss_step.append(round(unreplicate(train_metrics)['loss'].item(), 3))

            # Update Gradients
            if self.private and self.poisson_sampling:
                if self.train_data_loader.batch_sampler.end_of_batch:
                    lot_size = jax.device_put_replicated(self.lot_size, jax.local_devices())
                    state, dropout_rngs = parallel_optimizer_step(state, dropout_rngs, lot_size)
            elif self.private and not self.poisson_sampling:
                if (idx + 1) % self.gradient_accumulation_steps == 0 or len(self.train_data_loader) == (idx + 1):
                    accumulated_batch_size = self.physical_batch_size * self.gradient_accumulation_steps
                    accumulated_batch_size = jax.device_put_replicated(accumulated_batch_size, jax.local_devices())
                    state, dropout_rngs = parallel_optimizer_step(state, dropout_rngs, accumulated_batch_size)
            else:
                if (idx + 1) % self.gradient_accumulation_steps == 0 or len(self.train_data_loader) == (idx + 1):
                    state, dropout_rngs = parallel_optimizer_step(state, dropout_rngs)
        return state, train_metrics, dropout_rngs

    def evaluate(self, state, parallel_eval_step, with_generate=False):
        eval_metrics = None
        epoch_predictions = []
        all_labels = []
        for index, batch in tqdm(enumerate(self.eval_data_loader if not self.test else self.test_data_loader)):
            labels = batch["labels"]
            if with_generate:
                if self.generate:
                    predictions = parallel_eval_step(None, batch)
                else:
                    predictions = parallel_eval_step(state, batch)
            else:
                predictions, eval_metrics = parallel_eval_step(state, batch)
                self.validation_loss_step.append(round(unreplicate(eval_metrics)['loss'].item(), 3))

            predictions = [x for sublist in predictions for x in sublist]
            labels = [x for sublist in labels for x in sublist]

            if index == (len(self.eval_data_loader if not self.test else self.test_data_loader) - 1) and \
                    self.last_eval_batch_length != 0:
                predictions = predictions[:self.last_eval_batch_length]
                labels = labels[:self.last_eval_batch_length]

            decoded_predictions, decoded_labels = decode_postprocess_text(self.tokenizer, predictions, labels)
            if index == 0:
                inpt = batch['input_ids'][0][0]
                logger.info(f"MAX SEQUENCE LENGTH: {self.max_seq_len}")
                logger.info(f"BATCH SIZE: {self.batch_size_eval}")
                logger.info(f"INPUT: {self.tokenizer.decode(inpt, skip_special_tokens=True)}")
                logger.info(f"PREDICTION: {decoded_predictions[0]}")
                logger.info(f"LABEL: {decoded_labels[0][0]}")

            self.metrics.add_batch(predictions=decoded_predictions, references=decoded_labels)
            epoch_predictions += decoded_predictions
            all_labels += [ele for sub_list in decoded_labels for ele in sub_list]

        self.predictions = epoch_predictions
        self.references = all_labels
        return eval_metrics

    def update_grad_dp(self, state, dropout_rng, lot_size):
        grads_flat, grads_tree_def = jax.tree_util.tree_flatten(state.accum_grads)
        new_dropout_rng, *rngs = jax.random.split(dropout_rng, len(grads_flat) + 1)
        noise_std = self.noise_multiplier * self.l2_norm_clip
        noised = [(g + noise_std * jax.random.normal(r, g.shape, g.dtype)) / lot_size
                  for g, r in zip(grads_flat, rngs)]
        grad = jax.tree_util.tree_unflatten(grads_tree_def, noised)
        # Sync grad across devices by summing
        grad = jax.lax.psum(grad, "batch")
        state = state.apply_gradients(grads=grad)
        state = state.replace(accum_grads=jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, jnp.float32), state.params))
        return state, new_dropout_rng

    def update_grad(self, state, dropout_rng):
        grad = jax.tree_util.tree_map(lambda x: x / self.gradient_accumulation_steps, state.accum_grads)
        # Sync grad across devices by averaging
        grad = jax.lax.pmean(grad, "batch")
        state = state.apply_gradients(grads=grad)
        state = state.replace(accum_grads=jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape, jnp.float32), state.params))
        return state, dropout_rng

    def eval_step(self, state, batch):
        def eval_function(input_logits):
            return input_logits.argmax(-1)

        labels = batch.pop("labels")
        logits = state.apply_fn(**batch, params=state.params, train=False)[0]
        loss = self.loss_function(logits, labels)
        # Average the loss
        loss = jnp.mean(loss)
        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
        return eval_function(logits), metrics

    def generate_step(self, state, batch):
        pass

    def loss_function(self, logits, labels):
        cross_entropy = optax.softmax_cross_entropy(
            logits,
            jax.nn.one_hot(labels, num_classes=self.model.config.vocab_size))
        return jnp.mean(cross_entropy)

    def train_dp_step(self, state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def calc_loss_function(params, input_batch):
            targets = input_batch.pop("labels")
            logits = state.apply_fn(**input_batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            output_loss = self.loss_function(logits, targets)
            return output_loss

        grad_fn = jax.value_and_grad(calc_loss_function)

        # Insert dummy dimension in axis 1 to use jax.vmap over the batch
        batch = jax.tree_util.tree_map(lambda x: x[:, None], batch)
        # Use jax.vmap across the batch to extract per-example gradients
        grad_fn = jax.vmap(grad_fn, in_axes=(None, 0))

        loss, grad = grad_fn(state.params, batch)
        # Average the loss
        loss = jnp.mean(loss)

        # This step sums up the gradient along the batch/lot axis
        # (lot_size x A x B) -> (A x B)
        grads_flat, grads_tree_def = jax.tree_util.tree_flatten(grad)
        clipped_sum, _ = optax.per_example_global_norm_clip(grads_flat, self.l2_norm_clip)
        grad = jax.tree_util.tree_unflatten(grads_tree_def, clipped_sum)

        # Accumulate gradient
        state = state.replace(accum_grads=jax.tree_util.tree_map(lambda x, y: x + y, state.accum_grads, grad))

        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
        return state, metrics, new_dropout_rng

    def train_step(self, state, batch, dropout_rng):
        dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

        def calc_loss_function(params, input_batch):
            targets = input_batch.pop("labels")
            logits = state.apply_fn(**input_batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            output_loss = self.loss_function(logits, targets)
            return output_loss

        grad_fn = jax.value_and_grad(calc_loss_function)
        loss, grad = grad_fn(state.params, batch)
        # Average the loss
        loss = jnp.mean(loss)

        # Accumulate gradient
        state = state.replace(accum_grads=jax.tree_util.tree_map(lambda x, y: x + y, state.accum_grads, grad))

        metrics = jax.lax.pmean({"loss": loss}, axis_name="batch")
        return state, metrics, new_dropout_rng

    def _initialize_state(self):
        rng = jax.random.PRNGKey(self.seed)
        dropout_rngs = jax.random.split(rng, self.device_count)

        state = TrainState.create(
            apply_fn=self.model.__call__,
            params=self.model.params,
            tx=self.optimizer,
            accum_grads=jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, jnp.float32), self.model.params)
        )
        return state, dropout_rngs, rng

    def run_experiment(self):
        state, dropout_rngs, rng = self._initialize_state()
        parallel_eval_step = jax.pmap(self.eval_step, axis_name="batch")
        parallel_generate_step = jax.pmap(self.generate_step, axis_name="batch")
        if self.private:
            parallel_train_step = jax.pmap(self.train_dp_step, axis_name="batch", donate_argnums=(0,))
            parallel_optimizer_step = jax.pmap(self.update_grad_dp, axis_name="batch")
        else:
            parallel_train_step = jax.pmap(self.train_step, axis_name="batch", donate_argnums=(0,))
            parallel_optimizer_step = jax.pmap(self.update_grad, axis_name="batch")
        # Copy state on all devices
        state = flax.jax_utils.replicate(state)

        early_stop = early_stopping.EarlyStopping(
            min_delta=self.early_stop_min_delta,
            patience=self.patience
        )

        # Training Loop
        for i, epoch in enumerate(
                tqdm(range(1 + self.resume_from_epoch, self.epochs + self.resume_from_epoch + 1), desc="Epoch...")
        ):
            rng, input_rng = jax.random.split(rng)

            # train
            logger.info(f"Start training in epoch: {epoch}")
            current_time = datetime.now()
            self.start_time = current_time.strftime("%Y_%m_%d-%H_%M_%S")
            state, train_metrics, dropout_rngs = self.train(
                parallel_train_step,
                state,
                dropout_rngs,
                parallel_optimizer_step
            )
            # evaluate
            logger.info(f"Start evaluating in epoch: {epoch}")
            eval_metrics = self.evaluate(state, parallel_eval_step)
            result: Any = self.metrics.compute()
            self.result = {"bleu": result["score"]}
            train_loss = round(unreplicate(train_metrics)['loss'].item(), 3)
            metric_name = list(result.keys())[0]
            eval_loss = round(unreplicate(eval_metrics)['loss'].item(), 3)

            logger.info(f"{i + 1}/{self.epochs} | Train loss: {train_loss} | \
                Eval {metric_name}: {self.result} | Eval loss: {eval_loss}")

            self.training_loss_epoch.append(train_loss)
            self.validation_loss_epoch.append(eval_loss)

            plot_learning_curve(
                self.training_loss_step,
                self.validation_loss_step,
                self.write_dir,
                "learning_rate_plot_epoch_" + str(epoch) + ".png"
            )

            # Save model
            self.save_model(state)
            self.write_predictions(f"epoch_{epoch}")

            # reset metrics
            self.metrics = evaluate.load("sacrebleu")
            self.validation_loss_step = []

            if self.early_stopping:
                _, early_stop = early_stop.update(eval_loss)
                logger.info(f'Patience: {early_stop.patience_count}')
                if early_stop.should_stop:
                    logger.warning('Met early stopping criteria, breaking...')
                    self.early_stop_epoch = epoch
                    break

        if self.private:
            if self.early_stopping and self.early_stop_epoch is not None:
                actual_epochs = self.early_stop_epoch
            else:
                actual_epochs = self.epochs + self.resume_from_epoch
            num_examples = self.total_batch_size * len(self.train_data_loader)
            self.epsilon = compute_epsilons(
                num_examples,
                self.total_batch_size * self.gradient_accumulation_steps,
                self.noise_multiplier,
                actual_epochs
            )
            logger.info(f"Training finished. Epsilon: {self.epsilon}")
        else:
            logger.info("Training finished. No epsilon")

        # Save model
        self.save_model(state)

        # Save plots overall epochs
        plot_learning_curve(
            self.training_loss_epoch,
            self.validation_loss_epoch,
            self.write_dir,
            "learning_rate_plot_full_training.png",
            combined_plot=True
        )
        # evaluation metrics with generate
        logger.info(f"Generation step")
        self.evaluate(state, parallel_generate_step, with_generate=True)
        result: Any = self.metrics.compute()
        self.result = {"bleu": result["score"]}
        logger.info(f"Generation Eval BLEU: {self.result['bleu']}")
        # Log final results
        self.write_predictions()

    def run_generate(self):
        self.write_dir = self.experiment_path
        logger.info(f"Generation step")
        parallel_generate_step = jax.pmap(self.generate_step, axis_name="batch")
        self.evaluate(None, parallel_generate_step, with_generate=True)
        result: Any = self.metrics.compute()
        self.result = {"bleu": result["score"]}
        logger.info(f"Generation Eval BLEU: {self.result['bleu']}")
        # Log final results
        if self.test:
            self.write_predictions("final_step_test_set")
        else:
            self.write_predictions()
        plot_learning_curve(
            self.training_loss_epoch,
            self.validation_loss_epoch,
            self.write_dir,
            "learning_rate_plot_full_training.png",
            combined_plot=True
        )

    def save_model(self, state):
        self.model.save_pretrained(
            f"{self.write_dir}/{self.model_name}",
            params=jax.device_get(jax.tree_map(lambda x: x[0], state.params))
        )
        logger.info(f"Saved model to {self.write_dir}")

    def write_predictions(self, step="final_step"):
        file_path = f"result_{step}.json"
        data_to_write = {}

        jnp.save(
            f'{self.write_dir}/train_loss_{step}.npy',
            self.training_loss_step,
            allow_pickle=False
        )

        data_to_write["start_time"] = self.start_time
        data_to_write["end_time"] = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        data_to_write["num_devices"] = jax.local_device_count()
        data_to_write["total_batch_size_train"] = self.total_batch_size
        data_to_write["train_loss"] = self.training_loss_epoch
        data_to_write["eval_loss"] = self.validation_loss_epoch
        data_to_write["early_stop_epoch"] = self.early_stop_epoch
        if self.private:
            data_to_write["epsilon"] = self.epsilon

        data_to_write |= self.result
        data_to_write["predictions"] = self.predictions
        data_to_write["references"] = self.references

        with open(f"{self.write_dir}/{file_path}", "w") as file:
            json.dump(data_to_write, file, indent=4, separators=(',', ': '))

        logger.info(f"Write result file to: {self.write_dir}/{file_path}")


class MBartExperiment(Experiment):
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        # facebook/mbart-large-cc25 -> mbart-large-cc25
        self.model_name = settings.model.split('/')[-1]

    def __call__(self):
        # MBart specific tokenizer https://huggingface.co/docs/transformers/model_doc/mbart
        source_lang, target_lang = self.settings.lang_pair.split('-')

        source_lang = MBartExperiment.prepare_mbart_tokenizer(source_lang)
        target_lang = MBartExperiment.prepare_mbart_tokenizer(target_lang)

        self.tokenizer = MBartTokenizer.from_pretrained(
            "facebook/mbart-large-cc25",
            src_lang=source_lang,
            tgt_lang=target_lang
        )

        self.decoder_start_token_id = self.tokenizer.lang_code_to_id[target_lang]

        self.preprocessor = MBartPreprocessor(
            model=self.model,
            tokenizer=self.tokenizer,
            lang_pair=self.settings.lang_pair,
            max_seq_len=self.settings.max_seq_len
        )
        super()._setup_dataloader(self.settings)
        if not self.settings.generate:
            self._setup_optimizer(self.settings)

    def _setup_optimizer(self, settings):
        super()._setup_optimizer(settings)

        if settings.freeze:
            params = self.model.params
            params = freeze(params)
            partition_optimizers: Any = {
                'trainable': self.optimizer,
                'frozen': optax.set_to_zero()
            }

            def freeze_params(path, v):
                if 'encoder_attn' in path or 'encoder_attn_layer_norm' in path:
                    return 'trainable'
                else:
                    return 'frozen'

            param_partitions = traverse_util.path_aware_map(freeze_params, params)
            self.optimizer = optax.multi_transform(partition_optimizers, param_partitions)

    def generate_step(self, state, batch):
        _ = batch.pop("labels")
        if state is not None and not self.generate:
            self.model.params = state.params
        output_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=512,
            num_beams=self.num_beams,
            decoder_start_token_id=self.model.config.pad_token_id,
        )
        return output_ids.sequences

    @staticmethod
    def prepare_mbart_tokenizer(lang):
        xx_lang = ['en', 'es', 'fr', 'ja', 'nl']
        special_lang = {
            "cs": "CZ",
            "et": "EE",
            "gu": "IN",
            "hi": "IN",
            "kk": "KZ",
            "ko": "KR",
            "my": "MM",
            "ne": "NP",
            "si": "LK",
            "zh": "CN"
        }

        if lang in xx_lang:
            lang = lang + '_XX'
        elif lang in special_lang.keys():
            lang = lang + "_" + special_lang[lang]
        else:
            lang = lang + "_" + lang.upper()

        return lang


class T5Experiment(Experiment):
    def __init__(self, settings):
        super().__init__(settings)
        self.settings = settings
        # checkpoint/t5-small -> t5-small
        self.model_name = settings.model.split('/')[-1]

    def __call__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_name}")

        self.preprocessor = T5Preprocessor(
            model=self.model,
            tokenizer=self.tokenizer,
            lang_pair=self.settings.lang_pair,
            max_seq_len=self.settings.max_seq_len
        )

        super()._setup_dataloader(self.settings)
        if not self.settings.generate:
            self._setup_optimizer(self.settings)

    def _setup_optimizer(self, settings):
        super()._setup_optimizer(settings)

        if settings.freeze:
            params = self.model.params

            def freeze_params(path, v):
                if 'EncDecAttention' in path:
                    return v
                elif 'layer_norm' in path:
                    if '1' in path[-3:]:
                        return v
                    else:
                        return lax.stop_gradient(v)
                else:
                    return lax.stop_gradient(v)

            params = traverse_util.path_aware_map(freeze_params, params)
            self.model.params = params

    def generate_step(self, state, batch):
        _ = batch.pop("labels")
        if state is not None and not self.generate:
            self.model.params = state.params
        output_ids = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=512,
            num_beams=self.num_beams,
        )
        return output_ids.sequences


class MT5Experiment(T5Experiment):
    def __init__(self, settings):
        super().__init__(settings)

    def __call__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/{self.model_name}")

        self.preprocessor = MT5Preprocessor(
            model=self.model,
            tokenizer=self.tokenizer,
            lang_pair=self.settings.lang_pair,
            max_seq_len=self.settings.max_seq_len
        )

        super()._setup_dataloader(self.settings)
        if not self.settings.generate:
            super()._setup_optimizer(self.settings)
