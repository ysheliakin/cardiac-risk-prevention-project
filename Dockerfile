####
FROM ubuntu:18.04

#set timezone
ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

#update repositories
RUN apt-get -y update
RUN apt-get -y upgrade

#install pip3
RUN apt-get -y install python3-pip 

#install cython3
RUN apt-get -y install cython3

#install numpy, scipy, etc.
RUN apt-get -y install python3-numpy python3-pandas python3-sklearn python3-scipy python3-matplotlib jupyter-notebook

RUN pip3 install --no-cache-dir --upgrade --ignore-installed pip setuptools packaging

RUN pip3 install --no-cache-dir --ignore-installed \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    matplotlib \
    jupyter \
    imbalanced-learn \
    seaborn

#create a new user datascience
RUN useradd -ms /bin/bash datascience

#change user
USER datascience

#run jupyter notebook
WORKDIR /home/datascience
CMD jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
