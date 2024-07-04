# MFDNet: Multi-Frequency Deflare Network for Efficient Nighttime Flare Removal

Yiguo Jiang, Xuhang Chen , Chi-Man Pun📮 , Shuqiang Wang📮, Wei Feng (📮 Corresponding Author)

**University of Macau, SIAT CAS, Tianjin University**

In ***The Visual Computer***

## ⚙️ Usage
### Installation
```bash
git clone https://github.com/Jiang-maomao/flare-removal.git
cd flare-removal
```

### Training
To train a model with your own data/model, you can edit the ```config/config.py``` and run the following codes.

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
You can use the ```deflare.ipynb ```
### Evaluation
You can run the ```evaluate.py ```

# 💗 Acknowledgements
This work was supported in part by the Science and Technology Development Fund, Macau SAR, under Grants 0141/2023/RIA2 and 0193/2023/RIA3.
