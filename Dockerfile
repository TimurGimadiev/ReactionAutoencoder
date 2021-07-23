FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install python3.8 python3-pip git python3-gdbm -y
RUN pip3 install tqdm numpy scikit-learn ipython
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/TimurGimadiev/ReactionAutoencoder.git
RUN cd ReactionAutoencoder && pip3 install -U -e .
WORKDIR ReactionAutoencoder/Data



