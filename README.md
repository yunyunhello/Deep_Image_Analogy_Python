# Description

This is a project based on [caffe](http://caffe.berkeleyvision.org) and MSRA's [Deep Image Analogy](https://github.com/msracver/Deep-Image-Analogy/tree/linux)

I modified the project [Deep Image Analogy](https://github.com/msracver/Deep-Image-Analogy/tree/linux) from C version to python version(except photo transfer part). My work was included in ./deep_image_analogy/src/*

It seems that this work is not so meaningful to the real world, but I regardeded it as a personal project for me to have a taste of multiple technique skills, such as (py)caffe, deep learning, cuda, google cloud platform, django, git, etc.

Besides the technique skills, the soft skill -- problem solving was also trained through this project. I encountered a variety of problems and solved them one by one. For example, since this project needed a GPU to accelerate calculation while I did not have one and one NVIDIA GPU was quite expensive for me, but I did not give up and actively to seek a solution, and finally found Google Could Platform and rent a GPU(Tesla K80). These obstacles often appeared in my project, either big or small, I confronted them bravely.

Note: For the consistence of my repo name, I name this project "caffe" temporarily. After I finish it, I will change it.
In addition, I am a greener in github, if there is anything involved with your copyright, please tell me. I will delete it soon.
Thanks a lot!

# Enviroment
Hardware Enviroment(Google Cloud Platfrom):  
• n1-standard-4 (4 vCPUs, 15 GB memory);  
• 1 NVIDIA Tesla K80 (12 GB memory);  
• 50 GB disk.  

Software Enviroment(Google Cloud Platform):  
• 64-bit Ubuntu 16.04 Operating System;  
• NVIDIA Driver 396.36 + CUDA Tllkit 9.2.88 + cuDNN v7.1.4 for CUDA9.2 + pycuda;  
• pycaffe;  
• python 2.7;  
• openCV 3.0.

# Result
<div align=center><img src="https://github.com/yunyunhello/caffe/blob/master/deep_image_analogy/example/result.png"/></div>
Every two lines is a contrast combination. The first line is the result from original C version code, the second line is the result from my python code.     
About Run Time: 70s(original program) VS 170s(my program)  
From the result picture and running time, we can see that my code need improvement. 

# Acknowledgments
My codes acknowledge [Deep Image Analogy](https://github.com/msracver/Deep-Image-Analogy/tree/linux) and [caffe](http://caffe.berkeleyvision.org)


