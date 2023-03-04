# Heterogeneous Graph Pooling Neural Network for Sleep Stage Classification

![model_architecture](fig/model_arc.jpg)



## Requirements

- Python 3.7
- PyTorch 1.8.0
- PyTorch Geometric 1.7.0
- CUDA 10.1

#### Steps:

1. Install Pytorch
```shell
pip install torch==1.8.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Install torch_scatter
```shell
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.7-cp37-cp37m-linux_x86_64.whl
```
3. Install torch_sparse
```shell
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl
```
4. Install torch_geometric
```shell
pip install torch_geometric==1.7.0
```



## File Structure

```python
[<cwd>]
   |-----[<data_root>]
   |           |-------[<feature_root>]
   |           |               |--------subject1.npy
   |           |               |--------subject2.npy
   |           |               |--------    ...
   |           |-------[label]  # use expert1
   |           |          |----1_1.npy
   |           |          |----2_1.npy
   |           |          |----  ...
   |           |-------[<adj_mat_root>]
   |                           |--------subject_1_adj_mat.npy
   |                           |--------subject_2_adj_mat.npy
   |                           |--------          ...
   |-----[configs]
   |         |--------config.ini
   |-----*.py
```



## Training and Evaluation

Run this command for training and evaluation:

```train
python training.py -c ./configs/config.ini
```



## Results

Our model achieves the following performance on ISRUC-3:

| Accuracy | F1-score |
| :------: | :------: |
|  79.01%  |  77.02%  |



## Contributing

The code of this repository is released under the [MIT](https://github.com/zhouyh310/SleepHGNN/blob/main/LICENSE) license.

