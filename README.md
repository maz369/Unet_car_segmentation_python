# Unet_car_segmentation_python

This repository contains python implementation of U-net network for car segmentation in 2D (grayscale or RGB). The training and test dataset are from carvana Kaggel challenge (144 car images and 144 masks). The model has been tested on Linux and Windows 10.

This implementation is based on Keras with Tensorflow backend. It has been tested on Python 3.6, Anaconda (4.7). The uploaded code requires a GPU but if you do not have one, simply uninstall tensorflow-gpu (pip uninstall tensorflow-gpu), then install regular tensorflow (pip install tensorflow) and the code will work.

This can serve as an example for gaining experience with the followings:  

    Train a network using your own images  

    Learn the structure of Unet netwrok

    Test and fine tune a network to improve the segmentation task

    Perform 2D segmentation with other datasets

# Test

    Unzip data folder to where python codes are located
    Create a virtual environment with Anaconda: conda create -n DL_example python=3.6
    Activate the environment: activate DL_example
    Change directory to where you cloned the current files: cd ./where files are downloaded
    Install required libraries: pip install requirements.txt
    
    python main.py -im_h 64 -im_w 64 -batch_size=5 -epoch=50
    

The last command will run the network and save learning curve and example prediction result into the main folder. 
Weights will be also saved. Providing flags is not necessary. In case no flag is provided (python main.py), default values in the above command will be used to train the network. Higher resolution and larger number of epochs will improve accuracy of the result.

Note there are 2 folders, one for grayscale (grayscale) and the other for color (rgb) training. The network used in these 2 examples is the same and the only difference is the size of image (Grayscale or RGB) and the size of input and output tensor of the network. I thought it would be useful to have both examples for the same dataset to better understand the differences. I have made each folder self-sufficient and independent of the other one.

# Reference

Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." In International Conference on Medical image computing and computer-assisted intervention, pp. 234-241. Springer, Cham, 2015.
License.

# License
The code comes "AS IS" with no warranty of any kind. It can be used for any educational and research purpose. Feel free to modify and/or redistribute.
