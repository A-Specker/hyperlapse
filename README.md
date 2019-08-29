# Hyperlapse with some kind of stuff in it

This Readme is more for myself than others.

## Install & Configure for eclipse
File -> New  -> c++ Project -> ...
Project -> Properties -> c/c++ build --> Settings -> c++ Compiler -> _includepaths_ (see below)
__includepaths:__ pkg-config --cflags opencv	    --> copy each path seperatly(!) into eclipse without -I infront

Project -> Properties -> c/c++ build --> Settings -> c++ Linker 
__libs:__ pkg-config --libs opencv
all with __-L__ goes (without -L infront) to search path.
The entries with __-l__ are modules, include the ones you need (see OpenCV doc for each funcition to find out)

Dont forget to Alt+Enter (Project Properties) and add the includes in c++ build -> Settings
