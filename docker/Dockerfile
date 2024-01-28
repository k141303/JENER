FROM nvcr.io/nvidia/pytorch:22.11-py3
LABEL maintainer=k141303

RUN pip3 install --upgrade pip
RUN pip3 install \
    hydra-core==1.2.0 \
    tqdm==4.64.1 \
    transformers==4.30.0 \
    liat_ml_roberta \
    datasets==2.9.0 \
    tensorboard \
    torch_struct 
