# [MFDNet: Multi-Frequency Deflare Network for Efficient Nighttime Flare Removal](https://link.springer.com/article/10.1007/s00371-024-03540-x)

Yiguo Jiang, Xuhang Chen, Chi-Man Pun📮, Shuqiang Wang, Wei Feng (📮 Corresponding Author)

**University of Macau, SIAT CAS, Huizhou University, Tianjin University**

In ***The Visual Computer***

## ⚙️ Usage
### Installation
```bash
git clone https://github.com/Jiang-maomao/flare-removal.git
cd flare-removal
```

### Training
You may download the <a href="https://github.com/ykdai/Flare7K">Flare7K</a> dataset first. If you want to train a model with your own data/model, you can edit the ```config/config.py``` and run the following codes.

For single GPU training:
```bash
python train.py
```
For multiple GPUs training:
```bash
accelerate config
accelerate launch train.py
```
If you have difficulties on the usage of accelerate, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

### Inference
Please run the ```deflare.ipynb ``` 
You can use the model you trained on your dataset, or download the pre-trained models on the Flare7K dataset <a href="https://github.com/Jiang-maomao/flare-removal/releases/tag/checkpoint">here</a>.
### Evaluation
You can run the ```evaluate.py ```

# 💗 Acknowledgements
This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grants 0141/2023/RIA2 and 0193/2023/RIA3.

### 🛎 Citation
If you find our work helpful for your research, please cite:
```
@article{jiang2024mfdnet,
  title={Mfdnet: Multi-frequency deflare network for efficient nighttime flare removal},
  author={Jiang, Yiguo and Chen, Xuhang and Pun, Chi-Man and Wang, Shuqiang and Feng, Wei},
  journal={The Visual Computer},
  volume={40},
  number={11},
  pages={7575--7588},
  year={2024},
  publisher={Springer},
  doi = {10.1007/s00371-024-03540-x}
}
```
