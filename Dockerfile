# FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
# FROM nvcr.io/nvidia/pytorch:21.05-py3
# copy files
ADD scripts /workspace/
# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel einops tqdm opencv-python-headless
ENV CUDA_VISIBLE_DEVICES=0
