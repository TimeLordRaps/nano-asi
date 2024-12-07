FROM ghcr.io/open-webui/open-webui:cuda

# Update and install requirements for Mambaforge
RUN apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*

# Install Mambaforge for mamba
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O /tmp/mambaforge.sh \
    && bash /tmp/mambaforge.sh -b -p /opt/conda \
    && rm /tmp/mambaforge.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:$PATH"

# Create the nano-asi conda environment with CUDA-enabled PyTorch and xformers
RUN mamba create --name nano-asi python=3.10 pytorch-cuda=11.8 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y

# Install all required Python packages into nano-asi environment
# unsloth (from GitHub), nano-graphrag, distilabel, langgraph, mergekit, aider
RUN /opt/conda/envs/nano-asi/bin/pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    && /opt/conda/envs/nano-asi/bin/pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes \
    && /opt/conda/envs/nano-asi/bin/pip install nano-graphrag distilabel langgraph mergekit aider

ENTRYPOINT ["bash"]
