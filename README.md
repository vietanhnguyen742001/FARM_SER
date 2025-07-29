
## How To Use
- Clone this repository 
```bash
git clone https://github.com/vietanhnguyen742001/FARM_SER.git
cd MemoCMT
```
- Create a conda environment and install requirements
```bash
conda create -n MemoCMT python=3.8 -y
conda activate MemoCMT
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

- Dataset used in this project is [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) and [ESD](https://hltsingapore.github.io/ESD/). 

```bash
cd scripts && python preprocess.py -ds IEMOCAP --data_root ./data/IEMOCAP_full_release
```

- Before starting training, you need to modify the [config file](./src/configs/base.py) in the config folder. You can refer to the config file in the config folder for more details.

```bash
cd scripts && python train.py -cfg ../src/configs/hubert_base.py
```

- You can also find our pre-trained models in the [release].

## Citation
```bibtex

```
---

> GitHub [@vietanhnguyen742001
](vietanhnguyen742001
@gmail.com)
