FROM python:3.11

RUN git clone https://github.com/thesstefan/teshub
WORKDIR ./teshub/teshub/server

RUN pip install gdown

RUN gdown 1vE11lduBxK6nKxZVwJmGLiE0kq6j2cyd 
RUN gdown 1WAkKZH2VcBxKvSCEFHkI3buKEogGi3aL
RUN gdown 1AUyrGocZqGhR6d-cQltmn4cdjgugEdSW
RUN gdown 1HcWaihmp9zTevRHE6a_K5F3mp1bHLok1

RUN pip install "../../.[server]"

ENV SECRET_KEY=SECRET
ENV SEG_MODEL_CKPT=INFORMER.ckpt
ENV MORPH_ADD_SNOW_CKPT=ADD_SNOW.ckpt
ENV MORPH_ADD_FOG_CKPT=ADD_FOG.ckpt
ENV MORPH_ADD_CLOUDS_CKPT=ADD_SNOW.ckpt
