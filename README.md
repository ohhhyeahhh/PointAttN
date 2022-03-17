# PointAttN
### Installation

1. The project is implement in python 3.8.12, torch 1.9.0 and cuda 11.2

2. please Install the related dependencies before run the code:

+ h5py 3.6.0
+ matplotlib 3.4.3
+ munch 2.5.0
+ open3d 0.13.0
+ PyTorch 1.9.0
+ yaml 5.4.1

3. please compile the [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) for CD_Loss and compile the [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion) for other modules. Another way is using the following command:

```
cd $PointAttN_Home/utils/ChamferDistancePytorch/chamfer3D
python setup.py install

cd $PointAttN_Home/utils/mm3d_pn2
python setup.py build_ext --inplace
```



### Dataset

The benchmarks of [PCN](https://www.shapenet.org/) and [Compeletion3D](http://completion3d.stanford.edu/) are available below:

+ [PCN](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
+ [Completion3D](https://completion3d.stanford.edu/)



### Train

To train PointAttN, place the dataset with correct path and run train.py as the following command:

```
python train.py -c PointAttN.yaml
```



### Test

The pretrained models on Completion3D and PCN dataset are available as follows:

+ [PCN](https://pan.baidu.com/s/187GjKO2qEQFWlroG1Mma2g) (code：kmju)
+ [Completion3D](https://pan.baidu.com/s/17-BZr3QvHYjEVMjPuXHXTg) (code：nf0m)

To test PointAttN on PCN benchmark, place the pretrained model with correct path and run test_pcn.py as the following command:

```
python test_pcn.py -c PointAttN.yaml
```

To test PointAttN on Completion3D benchmark, place the pretrained model with correct path and run test_pcn.py as the following command:

```
python test_c3d.py -c PointAttN.yaml
```



### Cite this work

If you use PointAttN in your work, please cite our paper:

```
@article{PointAttN,
  title={PointAttN: You Only Need Attention for Point Cloud Completion},
  author={Wang, Jun and Cui, Ying and Guo, Dongyan and Li, Junxia and Liu, Qingshan, and Shen, Chunhua},
  journal={arXiv:2203.08485},
  year={2022}
}
```




## [Acknowledgement]

1. We include the following PyTorch 3rd-party libraries:  
   [1] [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
   [2] [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion)
   
2. Some of the code of this project is borrowed from [VRC-Net](https://github.com/paul007pl/MVP_Benchmark)  

