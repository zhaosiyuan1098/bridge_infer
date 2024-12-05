# 使用官方的 nvidia/cuda:11.8.0-devel-ubuntu22.04 镜像作为基础
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 设置环境变量，以便在 Docker 构建过程中不被提示输入
ENV DEBIAN_FRONTEND=noninteractive

# 接受构建时传入的代理参数
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

# 设置环境变量
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# 更新包列表并安装必要的软件包
RUN apt update && \
    apt install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        wget \
        build-essential \
        gdb \
        sudo \
        git \
        libreadline-dev \
        openssh-server \
        vim && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# 配置 SSH
RUN mkdir /var/run/sshd && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    sed -i 's/PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    ssh-keygen -A

# 创建用户 siyuan，设置密码并赋予高权限
RUN useradd -ms /bin/bash siyuan && \
    echo 'siyuan:password123' | chpasswd && \
    echo 'siyuan ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# 修改工作目录权限
RUN mkdir -p /workspace && chown siyuan:siyuan /workspace

# 切换到 siyuan 用户
USER siyuan
WORKDIR /workspace

# 确保 xmake 可执行文件目录在 PATH 中
ENV PATH="/home/siyuan/.local/bin:${PATH}"

# 安装 xmake
RUN curl -fsSL https://xmake.io/shget.text | bash


# 安装 Miniforge
RUN curl -L -o /home/siyuan/Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" && \
    bash /home/siyuan/Miniforge3.sh -b -p /home/siyuan/miniforge3 && \
    rm /home/siyuan/Miniforge3.sh

# 更新 PATH 环境变量
ENV PATH="/home/siyuan/miniforge3/bin:${PATH}"

ARG CACHEBUST=1

# 切换回 root 用户以启动 SSH 服务
USER root

ENV PATH="/usr/local/cuda/bin:${PATH}"

# 开放 SSH 端口
EXPOSE 22

# 启动 SSH 服务
CMD ["/usr/sbin/sshd", "-D"]