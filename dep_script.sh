#!/bin/bash

echo "==>dependencies setup for deep_q_rl"

echo "==>updating current package..."
sudo apt-get update

echo "==>installing OpenCV..."
sudo apt-get install python-opencv

echo "==>installing Matplotlib..."
sudo apt-get install python-matplotlib python-tk

echo "==>installing mongodb ..."
sudo apt-get install mongodb

echo "==>installing pymongo ..."
pip install --user --upgrade pymongo

echo "==>installing openai gym ..."
pip install --user gym[all]

echo "==>installing Theano ..."
# some dependencies ...
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
pip install --user --upgrade --no-deps git+git://github.com/Theano/Theano.git

echo "==>installing Lasagne ..."
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

echo "==>installing pymongo ..."
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

echo "==>installing Gym ..."
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Packages below this point require downloads. 
mkdir build
cd build

if [ ! -d "./pylearn2" ]
then
echo "==>installing Pylearn2 ..."
# dependencies...
sudo apt-get install libyaml-0-2 python-six
git clone git://github.com/lisa-lab/pylearn2
fi
cd ./pylearn2
python setup.py develop --user
cd ..



echo "==>All done!"
