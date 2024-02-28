# Hare: Highly Accelerated ReLU for Privacy Preserving Computation

## 作品特色

Hare是一种高效的隐私保护神经网络预测协议，能够使得任何一种基于卷积神经网络的云端机器学习模型在我们的协议下转换成不泄露客户端以及服务端隐私信息的隐私保护模型。相比于前人的工作，我们精心设计神经网络非线性层的协议，将密文计算和参数准备全部放到离线阶段进行，使得在线阶段ReLU的计算全部为轻量运算，在同样安全假设的条件下，大大地减少了预测所需时间



## 目录描述

* `FrontEnd/` 包含网页前端文件以及图片预处理程序
* `include/` 包含Cheetah的线性层协议
* `SCI` 包含CrypTFlow2的SCI库以及Cheetah和Hare的非线性层协议
* `networks/` 包含预先生成好的多种神经网络的C++代码
* `patch/` 包含对于部分依赖库的补丁
* `pretrained`，包含预处理好的模型和图片数据，以及图片预处理程序
* `credits/` 包含依赖库的许可
* `scripts/` 包含构建项目和运行程序的脚本文件
* `test`/ 包含测试正确率的脚本文件，以及图片预处理程序



## 环境需求

- openssl
- c++ compiler (>= 8.0 for the better performance on AVX512)
- cmake >= 3.13
- git
- make
- OpenMP (optional, only needed by CryptFlow2 for multi-threading)

目前已验证的编译环境

- Ubuntu 18.04 with gcc 7.5.0 Intel(R) Xeon(R), cmake 3.13
- Ubuntu 18.04 with gcc 7.5.0 Intel(R) Core(TM), cmake 3.13



## 项目编译

* 在`hare/`目录下运行 `bash scripts/build-deps.sh`
  * 此脚本会下载并编译emp-tool、emp-ot、Eigen、SEAL、zstd、hexl工具
  * 如果遇到网络不好的情况导致执行build-deps.sh时提示缺少某个库，可参考build-deps.sh手动git clone对应的库到/deps目录下，并在此执行 `bash scripts/build-deps.sh`

* 在`hare/`目录下运行 `bash scripts/build.sh`，将会在 `/build/bin` 目录下产生9个可执行文件
  * `resnet50-hare`
  * `sqnet-hare`
  * `densenet121-hare`
  * `resnet50-cheetah`
  * `sqnet-cheetah`
  * `densenet121-cheetah`
  * `resnet50-SCI_HE`
  * `sqnet-SCI_HE`
  * `densenet121-SCI_HE`
* 如果只想要编译某个可执行文件，可在 `build/` 目录下执行make xxx，如想编译resnet50-hare，可执行 `make resnet50-hare`



## 程序运行

* 可选择直接运行可执行文件，例如想要执行Hare的ResNet50预测
  * 在一个终端，执行 `cat pretrained/resnet50_model_scale12.inp | build/bin/resnet50-hare r=1`
  * 其中resnet50_model_scale12.inp是模型参数，r=1表示是服务端。这里省略了默认参数`k=12 ell=41 nt=4 ip=127.0.0.1 p=12345`，k=12表示定点数精度为12位，ell=41表示计算过程中位长为41，nt=4表示四线程运行，ip=127.0.0.1表示服务端地址为127.0.0.1，p=12345表示在服务端12345端口启动程序。这些参数可以在命令处进行调整
  * 在另外一个终端，执行`cat pretrained/resnet50_input_scale12_pred249.inp | build/bin/resnet50-hare r=2`
  * 其中resnet50_input_scale12_pred249.inp是经过图像预处理的图像数据，r=2表示是客户端，其余默认参数与服务端一致。

* 可以用脚本执行程序，例如想要执行Hare的ResNet50预测
  * 在一个终端，执行`bash scripts/run-server.sh hare resnet50`
  * 在另外一个终端，执行`bash scripts/run-client.sh hare resnet50`
  * `hare`可替换为`cheetah或SCI_HE`来执行Cheetah和CrypTFlow2协议
  * `resnet50`可替换为`sqnet或densenet121`来执行SqueezeNet和DenseNet121网络
  * k、ell、nt、ip、p可以在`scripts/common.sh`中进行修改



## 图片预处理

* 可参考`hare/requirements.txt`进行python库的安装，可执行`pip install -r requirements.txt`进行配置，版本不一定要和文件中一样，可自行选择
* 本项目实现神经网络预测时，需要先将图片转换成.inp文件，以数据流的形式输入到程序中。`pretrained/img_preprocess`中的python文件可以实现这一点。使用方法为`python ResNet50_img_preprocess_main.py <input_img_filename> <scale>`，scale表示定点数精度



## 前后端运行

* 首先需要确保能够运行图片预处理程序，可参考上面的"图片预处理"部分进行尝试运行
* 在`hare/FrontEnd/`目录下执行`python frontend.py`开启客户端，打开网页客户端
* 在`hare/`目录将服务端程序运行起来，如 `cat pretrained/resnet50_model_scale12.inp | build/bin/resnet50-hare r=1`。
* 在网页之中选择图片与协议进行预测，等待一段时间后即可得到输出。



## 测试

* 确保项目编译完成，可参考项目编译部分。确保tensorflow、scipy等库安装完成，可参考`hare/requirements.txt`进行配置
* 在`hare/test/`目录下，执行`./test_pre.sh`生成图片数据
* 打开一个终端，执行`./test_client.sh`，再打开另一个终端，执行`./test_server.sh`
* 如果提示无运行权限，可执行`chmod +x test_pre.sh`等命令来给予执行权限



## 注意事项

本项目的模型沿用Cheetah，其中，ResNet50的分类标签是从1到1000的，sqnet的分类标签是从0到999的，因此输出结果时，对于同一张图片，ResNet50的分类结果标签会比sqnet大1。

