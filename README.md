# EMG-to-QWERTY Decoding with BiLSTM and Data Augmentation

UCLA ECE C247 Final Project – Winter 2026  


---

# 1. Project Overview

This project investigates the problem of decoding typed characters directly from surface electromyography (sEMG) signals recorded from the wrist. The task is based on the **EMG2QWERTY dataset released by Meta Reality Labs**, which contains synchronized EMG signals and ground-truth keystrokes during touch typing.

We explore several neural network architectures and data processing techniques to improve decoding performance, including:

- BiLSTM
- GRU
- Transformer
- Data augmentation techniques
- Temporal downsampling
- Beam search decoding

Our final system uses a **BiLSTM-CTC model with data processing techniques and beam search decoding**, achieving a **test CER of 7.74%** on the single-user dataset (#89335547).

Pipeline of the system:
EMG Signal → Spectrogram → Neural Network → CTC Loss → Beam Search Decoder → Text Output


---

# 2. Setup

## Clone the repository
```shell
git clone https://github.com/kexin0712/emg2qwerty-project.git

cd emg2qwerty-project
```
---

## Install environment

Create the conda environment using the provided configuration:

```shell
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .
```
Alternatively, install dependencies with pip:
```shell
pip install -r requirements.txt
pip install -e .
```

---

## Download the dataset, extract, and symlink to ~/emg2qwerty/data

```shell
#Download the EMG2QWERTY dataset:
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz

#Then link the dataset to the project directory:
ln -s ~/emg2qwerty-data-2021-08 ./data
```

---

# 3. Training

All training commands are executed through the `emg2qwerty.train` module.

## Baseline (TDSConv)
```shell
python -m emg2qwerty.train\
    user=single_user\
    model=tds_conv_ctc\
    trainer.accelerator=gpu trainer.devices=1
```

---

## BiLSTM
```shell
python -m emg2qwerty.train\
    user=single_user\
    model=bilstm_ctc\
    trainer.accelerator=gpu trainer.devices=1
```
---

## BiLSTM + TDSConv
```shell
python -m emg2qwerty.train\
    user=single_user\
    model=bilstm_tdsconv_ctc\
    trainer.accelerator=gpu trainer.devices=1
```

---

## GRU

```shell
python -m emg2qwerty.train\
    user=single_user\
    model=gru_ctc\
    trainer.accelerator=gpu trainer.devices=1\
```

---

## Transformer

```shell
python -m emg2qwerty.train\
    user=single_user\
    model=transformer_ctc\
    trainer.accelerator=gpu trainer.devices=1\
```
---

## Best Model：Fine-Tuned BiLSTM
```shell
python -m emg2qwerty.train\
    user=single_user\
    model=bilstm_ctc\
    trainer.accelerator=gpu\
    trainer.devices=1\ 
    trainer.max_epochs=40\
    optimizer.lr=2.5e-4\
    module.lstm_hidden_size=448\
    module.output_dropout=0.25 module.lstm_dropout=0.15\
    +trainer.gradient_clip_val=1.0
```

---

# 4. BiLSTM model Training with different Data Processing Techniques 

Add the following commands after the command used for the Best Model (Fine-Tuned BiLSTM) to apply the corresbonding data processing technique.

---
## Weakened SpecAugment + Gaussian Noise 
```shell
    transforms.specaug.n_time_masks=2 \
    transforms.specaug.time_mask_param=15 \
    transforms.specaug.n_freq_masks=1 \
    transforms.specaug.freq_mask_param=3 \
    transforms.train.1='${gaussian_noise}'
```

---
## Magnitude Warping 
```shell
    transforms.train.4='${magnitude_warp}'
```

---
## Temporal Attenuation
```shell
    transforms.train.1='${temporal_attenuation}'
```
---

---
## Weakened SpecAugment + Gaussian Noise + Magnitude Warping
```shell
    transforms.specaug.n_time_masks=2 \
    transforms.specaug.time_mask_param=15 \
    transforms.specaug.n_freq_masks=1 \
    transforms.specaug.freq_mask_param=3 \
    transforms.train="[${to_tensor},${gaussian_noise},${select_channels},${band_rotation},${temporal_jitter},${magnitude_warping},${downsample},${logspec},${specaug}]"
```
---

---
## Weakened SpecAugment + Gaussian Noise + Temporal Attenuation
```shell
    transforms.specaug.n_time_masks=2 \
    transforms.specaug.time_mask_param=15 \
    transforms.specaug.n_freq_masks=1 \
    transforms.specaug.freq_mask_param=3 \
    transforms.train="[${to_tensor},${temporal_attenuation},${gaussian_noise},${select_channels},${band_rotation},${temporal_jitter},${downsample},${logspec},${specaug}]"
```
---

---
## Ablation Experiments on Sampling Rate

Modify the downsampling factor to control the sampling rate:
Example values: 1, 2, 3，4, 6.

```shell
    transforms.downsample.factor = <factor>
```
---

# 5. Testing

After training, models can be evaluated using greedy decoding or beam search decoding.

---

## Greedy Decoding


python -m emg2qwerty.train\
user=single_user\
model=bilstm_ctc \
checkpoint=<path_to_checkpoint>\
train=False\
decoder=ctc_greedy


---

## Beam Search Decoding


python -m emg2qwerty.train\
user=single_user\
model=bilstm_ctc \
checkpoint=<path_to_checkpoint>\
train=False\
decoder=ctc_beam


The beam search decoder uses a **6-gram character-level language model**, located in:


models/lm/


---

# 5. Final Model

Final configuration used in our report:

Model architecture:

BiLSTM

Data processing techniques:

- weakened SpecAugment  
- Gaussian noise  
- temporal attenuation  

Temporal downsampling factor:


2


Final performance:

| Metric | Value |
|------|------|
| Validation CER | 9.24% |
| Test CER | **7.74%** |

---

# 6. References

Viswanath Sivakumar et al.  
*emg2qwerty: A Large Dataset with Baselines for Touch Typing using Surface Electromyography*  
NeurIPS 2024.

Dataset and original repository:  
https://github.com/facebookresearch/emg2qwerty