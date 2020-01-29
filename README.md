# Hyperlapse with some kind of stabilization in it

This Readme is more for myself.

## Graph based stabilization
The idea is to stabilize video taken by mounted cameras (i.e., cameras with motion) by not taking every n-th frame, but selecting the next frame as a silimar frame. When moving the camera from front to left and back to front, only front frames are chosen.

Therefore, choose a frame skip(10), and a range (3), so for each frame a similarity score is calculated for the n+7th to n+13th frame. Those scores are weights for a dijkstra search at the end.

## Graph weights
The single weights are calculated by a mix of different measures:
- Sparse optical flow
- Dense optical flow
- Block matching
- Earth mover distance (on color)

The following optical flows were tested:
- tvl1, brox, farneback, lukas-kanade, sift, surf

## Install & Configure for eclipse
Only openCV is needed.

File -> New  -> c++ Project -> ...
Project -> Properties -> c/c++ build --> Settings -> c++ Compiler -> _includepaths_ (see below)
__includepaths:__ pkg-config --cflags opencv	    --> copy each path seperatly(!) into eclipse without -I infront

Project -> Properties -> c/c++ build --> Settings -> c++ Linker 
__libs:__ pkg-config --libs opencv
all with __-L__ goes (without -L infront) to search path.
The entries with __-l__ are modules, include the ones you need (see OpenCV doc for each funcition to find out)

Dont forget to Alt+Enter (Project Properties) and add the includes in c++ build -> Settings
