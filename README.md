# FM-RT

A pytorch implementation of [FM-RT](http://arxiv.org/abs/1911.02752):

Yang Liu, Xianzhuo Xia, Liang Chen, Xiangnan He, Carl Yang, Zibin Zheng. Certifiable Robustness to Discrete Adversarial Perturbations for Factorization Machines, in SIGIR 2020

**Unfinished** now! For reasons that remain unclear, metrics are lower. If you find the cause, point out please.
![](https://img-blog.csdnimg.cn/20200809132930240.png) 

## Files in the folder
- `src/`
    - `BaseModel.py`: The base class encapsulates device initialization, train & test tasks for rank, classification and regression.
    - `FMRT.py`: Derived from BaseModel. Include training and evaling methods.
    - `dataset.py`: Derived from `torch.util.data.Dataset`
        - `FeatureDataset`: Dataset for FM.
- `main.py`: You can look for descriptions of args with `-h` 
- `../benchmarks/datasets/`

## Acknowledgements

For FM, We use [torchfm](https://github.com/rixwew/pytorch-fm/) with few changes.  
