# Tree-distribution-shift: GroundingDINO + MM-Grounding-DINO + Plain-DETR
# Host: CUDA 11.6 (Tesla T4, driver 510.x), Ubuntu 20.04, Python 3.9
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV QT_QPA_PLATFORM=offscreen

# Python 3.9 via deadsnakes PPA (ubuntu20.04 ships 3.8 by default)
# Qt5 libs required by GroundingDINO / mmcv for headless use
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git wget ca-certificates \
    python3.9 python3.9-venv python3.9-dev \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx build-essential \
    libqt5core5a libqt5gui5 libqt5widgets5 libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN wget -q https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3.9 /tmp/get-pip.py
RUN pip install --upgrade pip "setuptools<70" wheel

# PyTorch 1.13 + CUDA 11.6
RUN pip install torch==1.13.1 torchvision==0.14.1 \
    --index-url https://download.pytorch.org/whl/cu116

RUN pip install "numpy<2"

# tree-distribution-shift pip package
ARG GH_TOKEN=""
RUN if [ -n "$GH_TOKEN" ]; then \
      pip install "tree-distribution-shift @ git+https://${GH_TOKEN}@github.com/aadityabuilds/tree-shift-package.git@main"; \
    else \
      pip install tree-distribution-shift; \
    fi

# Plain-DETR deps
RUN pip install pycocotools tqdm cython scipy wandb timm==0.4.5

# PyQt5 provides libQt5Core-*.so.5.15 required by GroundingDINO/mmcv
RUN pip install PyQt5

# transformers (before GroundingDINO so PyTorch backend is recognized)
RUN pip install "transformers>=4.30,<4.46"

# Project files
WORKDIR /workspace
COPY . /workspace

ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# GroundingDINO at /opt (survives volume mount over /workspace)
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /opt/GroundingDINO
WORKDIR /opt/GroundingDINO
RUN pip install --no-build-isolation -e .
WORKDIR /workspace

RUN pip install "numpy<2" --force-reinstall

# MMDetection (opencv-headless avoids Qt; mmcv will use it instead of opencv-python)
RUN pip install opencv-python-headless
RUN pip install mmengine
RUN pip install mmcv==2.0.1 \
    -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
RUN pip install mmdet==3.3.0

RUN pip install "numpy<2" --force-reinstall

# Verify installs
RUN python -c "import mmdet; from mmdet.apis import DetInferencer; print('mmdet OK')"
RUN python -c "import groundingdino; print('GroundingDINO OK')"
RUN python -c "import torch; import transformers; from transformers import BertModel; print('torch+bert OK')"

# Cache BERT weights
RUN python -c "from transformers import BertModel, BertTokenizer; \
    BertTokenizer.from_pretrained('bert-base-uncased'); \
    BertModel.from_pretrained('bert-base-uncased'); \
    print('bert-base-uncased cached')"

RUN git config --global --add safe.directory /workspace

# DinoV3: clone pinned upstream commit, then patch for Python 3.9
ARG DINOV3_COMMIT=31703e4cbf1ccb7c4a72daa1350405f86754b6d1
RUN git clone https://github.com/facebookresearch/dinov3.git /opt/dinov3 \
    && cd /opt/dinov3 \
    && git checkout ${DINOV3_COMMIT}
RUN python /workspace/scripts/patch_dinov3_py39.py /opt/dinov3

ENV PYTHONPATH=/workspace
ENV GROUNDING_DINO_DIR=/opt/GroundingDINO

# PyQt5's bundled Qt5 libs must be in LD_LIBRARY_PATH for GroundingDINO/mmcv
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.9/dist-packages/PyQt5/Qt5/lib:/usr/local/lib/python3.9/site-packages/PyQt5/Qt5/lib:${LD_LIBRARY_PATH}
