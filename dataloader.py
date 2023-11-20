from opacus.data_loader import DPDataLoader, dtype_safe, shape_safe, logger
from torch.utils.data import Sampler, Dataset, DataLoader, IterableDataset
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from utils import MODEL_PRIVATE_MAXIMUM_BATCH_SIZE
from torch.utils.data import default_collate
from typing import List, Optional
import math
import torch
import pdb


def wrap_collate_with_empty(collate_fn):
    """
    Wraps given collate function to handle empty batches.
    Args:
        collate_fn: collate function to wrap
    Returns:
        New collate function, which is equivalent to input ``collate_fn`` for non-empty
        batches and outputs dictionary with ``skip_batch`` as the only key if
        the input batch is of size 0
    """

    def collate(batch):
        if len(batch) > 0:
            return collate_fn(batch)
        else:
            return {'skip_batch': True}

    return collate


class CustomDPDataLoader(DataLoader):
    """
    Custom DPDataLoader that not requires splitting batches if larger than the maximum batch size.
    Forked from the original DPDataLoader in opacus.
    """

    def __init__(
            self,
            dataset: Dataset,
            *,
            lot_size: int,
            physical_batch_size: int,
            collate_fn: None,
            drop_last: bool = False,
            generator=None,
            distributed: bool = False,
            model_name: str = None,
            **kwargs,
    ):

        self.distributed = distributed

        self.lot_size = lot_size
        self.physical_batch_size = physical_batch_size

        batch_sampler = CustomDPSampler(
                num_samples=len(dataset),  # type: ignore[assignment, arg-type]
                lot_size=lot_size,
                physical_batch_size=physical_batch_size,
                generator=generator,
                model_name=model_name,
            )
        if collate_fn is None:
            collate_fn = default_collate

        if drop_last:
            logger.warning(
                "Ignoring drop_last as it is not compatible with DPDataLoader."
            )

        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=wrap_collate_with_empty(
                collate_fn=collate_fn,
            ),
            generator=generator,
            **kwargs,
        )

    @classmethod
    def from_data_loader(
            cls, data_loader: DataLoader, *, distributed: bool = False, generator=None, model_name=None, lot_size=None, physical_batch_size=None
    ):

        if isinstance(data_loader.dataset, IterableDataset):
            raise ValueError("Uniform sampling is not supported for IterableDataset")

        return cls(
            dataset=data_loader.dataset,
            lot_size=lot_size,
            physical_batch_size=physical_batch_size,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last,
            timeout=data_loader.timeout,
            worker_init_fn=data_loader.worker_init_fn,
            multiprocessing_context=data_loader.multiprocessing_context,
            generator=generator if generator else data_loader.generator,
            prefetch_factor=data_loader.prefetch_factor,
            persistent_workers=data_loader.persistent_workers,
            distributed=distributed,
            model_name=model_name,
        )


class CustomDPSampler(UniformWithReplacementSampler):
    def __init__(self, *, num_samples: int, lot_size: int, physical_batch_size: int, generator=None, steps=None, model_name: str = None):
        """
        Args:
            num_samples: Dataset size.
            lot_size: Expected number of samples to draw for each batch.
            physical_batch_size: Max number of examples to process at a time.
            sample_rate: Probability used in sampling (lot_size / num_samples).
            generator: Generator used in sampling.
            steps: Number of steps (iterations of the Sampler)
            model_name: Name of the model to be used for the batch size
        """
        sample_rate = lot_size / num_samples
        super().__init__(num_samples=num_samples, sample_rate=sample_rate, generator=generator)
        self.model_name = model_name

        self.lot_size = lot_size
        self.physical_batch_size = physical_batch_size
        self.end_of_batch = False
        self.num_physical_batches_per_lot = torch.inf

        if steps is not None:
            self.steps = steps
        else:
            self.steps = math.ceil(1 / self.sample_rate)

    def __len__(self):
        return self.steps

    def __iter__(self):
        num_total_batches = self.steps
        while num_total_batches > 0:
            mask = (
                torch.rand(self.num_samples, generator=self.generator)
                < self.sample_rate
                )
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()

            # There will be a remainder batch (depending on num data points 
            # sampled)
            self.num_physical_batches_per_lot =\
                math.ceil(len(indices) / self.physical_batch_size)

            for physical_batch_idx in range(self.num_physical_batches_per_lot):
                if physical_batch_idx == (self.num_physical_batches_per_lot - 1):
                    self.end_of_batch = True
                else:
                    self.end_of_batch = False
                start_idx = physical_batch_idx * self.physical_batch_size
                end_idx = (physical_batch_idx + 1) * self.physical_batch_size
                batch_indices = indices[start_idx:end_idx]
                yield batch_indices

            num_total_batches -= 1
