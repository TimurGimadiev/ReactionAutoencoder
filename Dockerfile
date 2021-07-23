FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install python3.9 python3-pip git ipython
RUN pip3 install -U git+https://github.com/TimurGimadiev/ReactionAutoencoder.git ipython tqdm numpy scikit-learn
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


