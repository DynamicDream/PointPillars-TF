# PointPillar with TF
先阅读完注意事项，各位再往下操作会容易很多.
# 注意事项：
```
   1.model.h5文件放到项目目录中新建的logs文件夹下
   2.在执行python setup.py install之前,需要将pybind11下载解压到项目中的pybind11文件夹中
   3.若执行python setup.py install时，报错未定义M_PI,此时,将src文件夹中的 point_pillars.cpp 打开,并预定义#define M_PI 3.1415926 即可
   4.执行python point_pillars_inference.py  --data_root=/media/data/kitti-3d/kitti/testing  --result_dir=./results/ --model_path=./logs/model.h5 时,
   可能报错，修改--data_root=/media/data/kitti-3d/kitti/testing 为 --data_root=./media/data/kitti-3d/kitti/testing
   5.测试集需要在项目目录下创建\media\data\kitti-3d\kitti\testing，并且按照
   └── testing     <-- 7580 test data
           ├── calib
           ├── velodyne
           ├── imag_2
    的形式创建
   6.若报错内容为zlibwapi.dll相关错误,请按照:https://blog.csdn.net/qq_45071353/article/details/124091856 解决
   7.若报错内容为Could not load library cudnn_cnn_infer64_8.dll.则查看是否CUDA版本为11.5,此时的cudNN不能用官方给的匹配版本,需要按照CUDA11.4的版本下载cudNN
```


### 环境安装

准备conda环境，推荐使用miniconda，新建conda环境：

```bash
conda create -n tensorflow python=3.8
conda activate tensorflow
conda install tensorflow-gpu
pip install tensorflow_probability==0.12.1 sklearn opencv-python easydict tqdm
```

### 数据准备：

```bash
├── training    <-- 7481 train data
   |   ├── calib
   |   ├── label_2
   |   ├── velodyne
└── testing     <-- 7580 test data
           ├── calib
           ├── velodyne
```


### 开始训练

```bash
python point_pillars_training_run.py --imageset_path=./image_sets/ --data_root=/media/data/kitti-3d/kitti/training/ --model_root=./logs/
```

一共训练160个epoch，训练完成后，运行推理：

```bash
python point_pillars_inference.py  --data_root=/media/data/kitti-3d/kitti/testing  --result_dir=./results/ --model_path=./logs/model.h5
```


# Codebase Information
The base code has been taken from [tyagi-iiitv/PointPillars](https://github.com/tyagi-iiitv/PointPillars) GitHub repository.
As I found some bugs in the original repository, and it was not being actively maintained, I decided to create an alternate repository.

Major changes in this repo compared to the original one:
 - Correct transformation of the KITTI GT from camera coordinate frame to LiDAR corodinate frame.
 - Minor changes during target creation.
 - Slight changes in the model convolution setup (conv-bn-relu instead of the original conv(with_bias)-relu-bn).
 - Complete overhaul of inference pipeline with functionality of dumping 3D BBs projected on the image and dumping of labels in the KITTI evaluation toolkit expected format.
 - Unit-tests for checking code functionality.

Please note that I have not been able to achieve the same performance as claimed in the paper.
I am still working on it. 

# About Point Pillars
Point Pillars is a very famous Deep Neural Network for 3D Object Detection for LiDAR point clouds. With the application of object detection on the LiDAR devices fitted in the self driving cars, Point Pillars focuse on fast inference ~50fps, which was magnitudes above as compared to other networks for 3D Object detection. In this repo, we are trying to develop point pillars in TensorFlow. [Here's](https://medium.com/@a_tyagi/pointpillars-3d-point-clouds-bounding-box-detection-and-tracking-pointnet-pointnet-lasernet-67e26116de5a?source=friends_link&sk=4a27f55f2cea645af39f72117984fd22) a good first post to familiarize yourself with Point Pillars. 

**Contributors are welcome to work on open issues and submit PRs. First time contributors are welcome and can pick up any "Good First Issues" to work on.**

# PointPillars in TensorFlow
Point PIllars 3D detection network implementation in Tensorflow. External contributions are welcome, please fork this repo and see the issues for possible improvements in the code.  

# Installation
Download the LiDAR, Calibration and Label_2 **zip** files from the [Kitti dataset link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and unzip the files, giving the following directory structure:

```plain
├── training    <-- 7481 train data
   |   ├── calib
   |   ├── label_2
   |   ├── velodyne
└── testing     <-- 7580 test data
           ├── calib
           ├── velodyne
```
After placing the Kitti dataset in the root directory, run the following code

```
git clone --recurse-submodules https://github.com/viplix3/PointPillars-TF.git
virtualenv --python=/usr/bin/python3.8 env
source ./env/bin/activate
pip install tensorflow-gpu tensorflow_probability sklearn opencv-python
cd PointPillars
python setup.py install
python point_pillars_training_run.py
```

# Deploy on a cloud notebook instance (Amazon SageMaker etc.)
Please read this blog article: https://link.medium.com/TVNzx03En8

# Technical details about this code
Please refer to this [article](https://medium.com/@a_tyagi/implementing-point-pillars-in-tensorflow-c38d10e9286?source=friends_link&sk=90995fae2d0a9c4e0dd5ec420c218c84) on Medium. 

# Pretrained Model
The Pretrained Point Pillars for Kitti with complete training and validation logs can be accessed with this [link](https://drive.google.com/file/d/1VfnYr3N7gZb2RuzQNCTrTIZoaoLEzc8O/view?usp=sharing). Use the file model.h5.

# Saving the model as .pb
Inside the point_pillars_training_run.py file, change the code as follows to save the model in .pb format. 

```
import sys
if __name__ == "__main__":

    params = Parameters()

    pillar_net = build_point_pillar_graph(params)
    pillar_net.load_weights(os.path.join(MODEL_ROOT, "model.h5"))
    pillar_net.save('new_model')
    sys.exit()
    # This saves the model as pb in the new_model directory. 
    # Remove these lines during usual training. 
```
# Loading the saved pb model
```
model = tf.saved_model.load('model_directory')
```
