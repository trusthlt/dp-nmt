# DP-NMT: Scalable Differentially-Private Machine Translation

## Description

DP-NMT is a framework for carrying out research on privacy-preserving neural machine translation (NMT) with [differentially private stochastic gradient descent (DP-SGD)](https://arxiv.org/abs/1607.00133). Implemented using the [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax) libraries, DP-NMT brings together numerous models, datasets and evaluation metrics in one software package. Our goal is to provide a platform for researchers to advance the development of privacy-preserving NMT systems, keeping the details of the DP-SGD algorithm (e.g. Poisson sampling) transparent and intuitive to implement. We provide tools for training text generation models on both out-of-the-box and custom datasets, with and without differential privacy guarantees, using different sampling procedures for iterating over training data.

## Installation

With [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/):

```bash
conda create -n dp-nmt python=3.10
conda activate dp-nmt
git clone https://github.com/trusthlt/dp-nmt.git
cd dp-nmt
pip install -r requirements.txt
conda install cuda-nvcc -c conda-forge -c nvidia
```

## 1. Normal Training Example

### 1.1. With WMT16 dataset

```python
python main.py \
  --dataset wmt16 \
  --model google/mt5-small \
  --epochs 16 \
  --batch_size 16 \
  --optimizer Adam \
  --learning_rate 0.001 \
  --gradient_accumulation_steps 1 \
```

### 1.2. With custom dataset

```python
python main.py \
  --dataset data/bsd.py \
  --model google/mt5-small \
  --epochs 16 \
  --batch_size 16 \
  --optimizer Adam \
  --learning_rate 0.001 \
  --gradient_accumulation_steps 1 \
```

### 1.3 To continue training from a checkpoint

Keep the same arguments as in previous training and add the following
`--resume_from_epoch` to specify the epoch checkpoint to resume from and modify the checkpoint path `--model`.

```python

python main.py \
  --dataset wmt16 \
  --model checkpoints/2023_06_23-13_21_46/mt5-small \
  --epochs 16 \
  --batch_size 16 \
  --optimizer Adam \
  --learning_rate 0.001 \
  --gradient_accumulation_steps 1 \
  --resume_from_epoch 16  
```

## 2. Private Training Example

### 2.1. With Poisson sampling

```python
python main.py \
  --dataset wmt16 \
  --model google/mt5-small \
  --epochs 7 \
  --batch_size 16 \
  --lot_size 524288 \
  --optimizer Adam \
  --learning_rate 0.001 \
  --private True \
  --noise_multiplier 13.18 \
  --warmup_steps 4 \
  --custom_dp_dataloader True \
  --poisson_sampling True \
```

Batches drawn with Poisson sampling are determined based on the `--lot_size` argument and built up through gradient accumulation using a physical batch size of `--batch_size`. When `--poisson_sampling` is set to `True`, the argument `--gradient_accumulation_steps` is therefore not used.

### 2.2. Without Poisson sampling

```python
python main.py \
  --dataset wmt16 \
  --model google/mt5-small \
  --epochs 7 \
  --batch_size 16 \
  --optimizer Adam \
  --learning_rate 0.001 \
  --gradient_accumulation_steps 32768 \
  --private True \
  --noise_multiplier 13.18 \
  --warmup_steps 4 \
  --custom_dp_dataloader True \
  --poisson_sampling False \
```

Continue training and custom dataset are the same as in normal training.

## 3. Evaluation

By default, the evaluation while training without teacher forcing is done on the dev set using SacreBLEU. If the training 
is stopped before the end of the last epoch, the evaluation needs to be done separately. To do so, use the arguments
`--generate True` with `--resume_from_epoch` from the stopped epoch to plot the training and validation loss curves. 
Make sure to point to the correct checkpoint path `--model`.

```python
python main.py \
  --dataset wmt16 \
  --model checkpoints/2023_06_25-23_50_22/mt5-small \
  --resume_from_epoch 25 \
  --generate True \
```

Evaluation on the test set is similar and can be done using the argument `--test True`

```python
python main.py \
  --dataset wmt16 \
  --model checkpoints/2023_06_25-23_50_22/mt5-small \
  --resume_from_epoch 25 \
  --generate True \
  --test True \
```

Name of the output file is `result_final_step_test_set.json` and is saved in the checkpoint folder.

BERTScore can be used instead of SacreBLEU by in `evaluate_output.py` script. E.g.,
    
```python
python evaluate_output.py --data checkpoints/2023_06_25-23_50_22/result_final_step.json
```

Name of the output file is `result_final_step_bertscore.json` and is saved in the checkpoint folder. Similar to the test set.

## 4. Compute Epsilon

Example script to compute the epsilon value **without Poisson sampling**:

```python
python compute_epsilons.py
    --dataset wmt16 \
    --lang_pair de-en \
    --batch_size 16 \
    --gradient_accumulation_steps 32768 \
    --device_count 2 \
    --epochs 25
```

**With Poisson sampling**, always set `--gradient_accumulation_steps` to 1 and `--device_count` to 1. Set `--batch_size` to the lot size that will be used during training. For example:

```python
python compute_epsilons.py
    --dataset wmt16 \
    --lang_pair de-en \
    --batch_size 524288 \
    --gradient_accumulation_steps 1 \
    --device_count 1 \
    --epochs 25
```

If `--noise_multiplier` is not specified, the script will compute the epsilon for a range of noise multipliers.
Otherwise, it will compute the epsilon for the specified noise multiplier.
