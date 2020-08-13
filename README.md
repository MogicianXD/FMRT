# FM-RT

A pytorch implementation of [FM-RT](http://arxiv.org/abs/1911.02752):

Yang Liu, Xianzhuo Xia, Liang Chen, Xiangnan He, Carl Yang, Zibin Zheng. Certifiable Robustness to Discrete Adversarial Perturbations for Factorization Machines, in SIGIR 2020


## Note

The paper used a FM as: 

![](https://img-blog.csdnimg.cn/20200813213411248.png#pic_center)

`FMRT2.py` is the corresponding implementation.

However, I cannot see similar metrics. Before FMRT, the avg-max $q$ is only `0.88`, while the paper showed `1.60`.  

So I followed the formula Rendle adopted, which is

![](https://img-blog.csdnimg.cn/20200813213353416.png#pic_center)

Then I use `FMRT.py` by default. It dismissed elements on the diagonal.

Here's the performance comparison:
![](https://img-blog.csdnimg.cn/20200813210408734.png) 

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
