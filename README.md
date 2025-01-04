# MIRRN

KDD2025 ! ! ! 

This repository is the implementation for Paper "Multi-granularity Interest Retrieval and Refinement Network for Long-Term User Behavior Modeling in CTR Prediction".




## Requirements

* Ensure you have Python and PyTorch (version 1.8 or higher) installed. Our setup utilized Python 3.8 and PyTorch 1.13.0.
* Should you wish to leverage GPU processing, please install CUDA.



## Dataset

We use three public real-world datasets (Taobao, Alipay and Tmall) in our experiments. We pre-process the data in the same way with ETA and SDIM. You can download the datasets from the links below.

- **Taobao**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=649. If you want to know how to preprocess the data, please refer to `./data/taobao/preprocess.py`
- **Alipay**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=53. If you want to know how to preprocess the data, please refer to `./data/alipay/preprocess.py`
- **Tmall**: The raw dataset can be downloaded from https://tianchi.aliyun.com/dataset/dataDetail?dataId=42. If you want to know how to preprocess the data, please refer to `./data/tmall/preprocess.py`




## Example

If you have downloaded the source codes, you can train MIRRN model. 

```
$ cd main
$ python build_taobao_to_parquet.py
$ python run_expid.py
```

You can change the model parameters in `./main/config/General_config/model_config.yaml`



## Contact

Should you have any questions regarding our paper or codes, please don't hesitate to reach out via email at [demon@mail.ustc.edu.cn](mailto:demon@mail.ustc.edu.cn).




## Acknowledgment 

Our code is developed based on [reczoo/FuxiCTR: A configurable, tunable, and reproducible library for CTR prediction https://fuxictr.github.io](https://github.com/reczoo/FuxiCTR).

