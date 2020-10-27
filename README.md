# Weakly Supervised Object Detection

This repository hosts a public version of the code used for my bachelor thesis ["Weakly Supervised Object Detection in RoboCup Scenarios"](https://github.com/Menkeyshow/Weakly-Supervised-Object-Detection/blob/main/paper/bachlor_thesis.pdf).
By using a convolutional neural network architecture proposed by Daniel Speck et al. in ["Ball localization for robocup soccer using convolutional neural networks"](https://link.springer.com/chapter/10.1007/978-3-319-68792-6_2) for the RoboCup context and utilizing only coarse level labels, one can obtain more detailed results by manipulating the predictions, than with more commonly used supervised learning methods.
The manipulations are mainly based on the paper [Simple Does It proposed by Anna Khoreva et al.](https://arxiv.org/abs/1603.07485) and the GrabCut algorithm, an image segmentation method based on graph cuts.

Additionally i have included an [earlier work](https://github.com/Menkeyshow/Weakly-Supervised-Object-Detection/blob/main/paper/multiobjectfcnn_Birkenhagen_Geislinger__Copy_.pdf) , which evaluates the influence of different parameters referenced in the bachelor thesis.



## Requirements
The code runs on python 3.x versions. 
- dependencies:
  - python=3.7
  - pydot
  - numpy
  - opencv
  - matplotlib
  - tensorflow
  - pylint
  - spyder==3.3.6
