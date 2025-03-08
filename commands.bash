docker run -it tensorflow/tensorflow bash

docker run -it --rm --runtime=nvidia tensorflow/tensorflow:latest-gpu python

docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-jupyter

docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py

# For virtual env
source /Library/Frameworks/Python.framework/Versions/3.11/bin/virtualenvwrapper.sh 

source ~/.storevirtualenvs/ai/bin/activate