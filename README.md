# Masterthesis

The git contains numourus samples of code used for developing the SLAM project. 

The final code for examination is the code in the Clean folder.

The code has 2 optional inputs:
skip: sets how many frames are skipped per iteration
display: enables the sdl2 2D Display

The path to the video for testing as well as the camera intrinsic matrix are hardcoded and needs to be altered depending on the use in the Clean/slam.py file

Example to run the code: skip=5 display=1 ./slam.py

Dependencies: 
python 3.7
numpy
opencv contrib version
sdl2 
pangolin, python bindings: https://github.com/uoip/pangolin
g2o, python bindings: https://github.com/uoip/g2opy
