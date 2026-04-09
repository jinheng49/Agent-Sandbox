# 使用官方的 Python 3.11 精简版镜像作为底座
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装基础的系统构建工具
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制当前目录的内容到容器的 /app 中
COPY . /app

# 升级 pip 并安装依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install langgraph langchain-core qdrant-client streamlit pydantic rich openai

# 暴露 Streamlit 的默认端口
EXPOSE 8501