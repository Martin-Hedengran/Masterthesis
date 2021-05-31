# Masterthesis

The git contains numourus samples of code used for developing the SLAM project. 

**The final code for examination is the code in the Clean folder.**

The code has 2 optional inputs:
- skip: Sets how many frames are skipped per iteration
- display: Enables the sdl2 2D display which shows feature extraction and matching

The path to the video for testing as well as the camera intrinsic matrix are hardcoded and needs to be altered depending on the use in the Clean/slam.py file

**Example to run the code from ./Clean folder:** 
```Bash
skip=5 display=1 ./slam.py
```
Dependencies: 
- python 3.7
- numpy
- opencv contrib version
- skimage
- sdl2 
- pangolin, python bindings: https://github.com/uoip/pangolin
- g2o, python bindings: https://github.com/uoip/g2opy
