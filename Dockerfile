# build based on nvidia pytorch container
FROM nvcr.io/nvidia/pytorch:18.08-py3
#RUN pip install --upgrade pip 
RUN pip install --upgrade pip cython
RUN pip install jupyter numpy matplotlib visdom rasterio pandas

COPY . /MTLCC-pytorch

ENV PYTHONPATH "${PYTONPATH}:/MTLCC-pytorch/src"

# set working directory (default entry directory)
WORKDIR /MTLCC-pytorch

