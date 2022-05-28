# e4040-2021Fall-project

## FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction

Reproduced and experimented on FishNet, disscussion of the results obtained. 

## Authors
* Yuqing Cao (yc3998)
* Fengyang Shang (fs2752)
* Chengbo Zang (cz2678)

## Description

With increasingly demanding tasks in computer vision problems, neural network models go deeper and are divided into image level, region level and pixel level tasks to deal with more complex situations. In this project, we reviewed the paper FishNet: A Versatile Backbone for Image, Region, and Pixel Level Predictions, which proposed a new model suitable for all level predictions. We reproduced the model and conducted several experiments under necessary modifications with Mnist, Cifar10 and Cifar100 datasets. It can be deduced that FishNets could achieve better classification accuracy with less computational cost. 

## Execution

* For the use of command line tools, run 
```
python fish.py fishnet55 cifar10 -e 30
```

* Or use the follwing command for detailed instructions. 
```
python fish.py -h
```

* To see development interface, open and execute blocks in [main.ipynb](https://github.com/ecbme4040/e4040-2021fall-project-fcsz-yc3998-fs2752-cz2678/blob/0f50dd3c837a922aef1f8a3e6096bb4cca834fd6/main.ipynb). 
* To see experiment interface, open and execute blocks in [network_graph.ipynb](https://github.com/ecbme4040/e4040-2021fall-project-fcsz-yc3998-fs2752-cz2678/blob/0f50dd3c837a922aef1f8a3e6096bb4cca834fd6/network_graph.ipynb). 


## Organization of this directory
```
.
├── README.md
├── fish.py                 : Command line tool
├── imgprocess.py           : Image preprocessing
├── main.ipynb              : Main development illustration
├── main.py                 : Train function definition
├── history                 : Training history
├── model                   : Model definition
│   ├── __init__.py
│   ├── applications.py
│   ├── blocks.py
│   └── fishnet.py
├── network_graph.ipynb     : Main experiments interface
└── train.sh                : Bash for command line training

1 directory, 11 files
```


## References

[1] Sun, S., Pang, J., Shi, J., Yi, S., & Ouyang, W. (2019). Fishnet: A versatile backbone for image, region, and pixel level prediction. 

[2] D.O. Hebb(1949) The Organization of Behavior: A Neuropsychological Theory.

[3] Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386–408. https://doi.org/10.1037/h0042519.

[4] A Krizhevsky, I Sutskever, GE Hinton. (2012) Imagenet classification with deep convolutional neural networks - Advances in neural information processing systems.

[5] He K., Zhang X., Ren S., Sun J. (2016) Identity Mappings in Deep Residual Networks. In: Leibe B., Matas J., Sebe N., Welling M. (eds) Computer Vision – ECCV 2016. ECCV 2016. Lecture Notes in Computer Science, vol 9908. Springer, Cham. https://doi-org.ezproxy.cul.columbia.edu/10.1007/978-3-319-46493-0_38.

[6] Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks. Science, 313(5786), 504–507. http://www.jstor.org/stable/3846811.

[7] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.

