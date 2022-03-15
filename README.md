# PointAttN
The project is implement in python=3.8.12, torch=1.9.0 and cuda=11.2

### Installation

Install dependencies:

+ h5py 3.6.0
+ matplotlib 3.4.3
+ munch 2.5.0
+ open3d 0.13.0
+ PyTorch 1.9.0
+ yaml 5.4.1

### Compile Pytorch 3rd-party modules

1. [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) or

cd ChamferDistancePytorch
python setup.py install

2. [emd, expansion_penalty](https://github.com/Colin97/MSN-Point-Cloud-Completion) or

cd EMD
python setup.py install
cd expansion_penalty
python setup.py install

3. [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion) or

cd mm3d_pn2
python setup.py install

### Train

python train.py -c PointAttN.yaml

### Test

python test_pcn.py -c PointAttN.yaml


### Dataset

The dataset used in our experiments are PCN and Compeletion3D and is available below:
[PCN](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
[Completion3D](https://completion3d.stanford.edu/)

The pretrained models on Completion3D and PCN dataset are available as follows:
[PCN](https://pan.baidu.com/s/187GjKO2qEQFWlroG1Mma2g) 提取码：kmju
[Completion3D](https://pan.baidu.com/s/17-BZr3QvHYjEVMjPuXHXTg) 提取码：nf0m


## [License]

Our code is released under MIT License.


## [Acknowledgement]

1. We include the following PyTorch 3rd-party libraries:  
   [1] [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
   [2] [emd, expansion_penalty, MDS](https://github.com/Colin97/MSN-Point-Cloud-Completion)  
   [3] [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)  

2. Some of the code of this project is borrowed from [VRC-Net](https://github.com/paul007pl/MVP_Benchmark)  

