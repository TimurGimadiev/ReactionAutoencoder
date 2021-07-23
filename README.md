### Reaction Autoencoder

Neural network architecture close to one published in *[article](https://doi.org/10.1038/s41598-021-81889-y)*


### Installation with virtualenv

    virtual -p python3.9 venv
    . venv/bin/activate
    git clone https://github.com/TimurGimadiev/ReactionAutoencoder.git
    cd ReactionAutoencoder
    pip install -U -e .

### Docker usage

Do not clone repository, just copy Dockerfile and do
    
    docker build . -t autoencoder

then run it with
    
    docker run --rm -it autoencoder python3

if you have nvidia support for docker
    
    docker run --gpus all --rm -it autoencoder python3

### Usage

please take a look at test.py 



