# Use AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Workdir (Lambda uses /var/task)
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Install minimal build tools and rust toolchain into temporary locations,
# install python deps, then remove rust/cargo and pip cache (but DO NOT remove yum).
ENV RUSTUP_HOME=/tmp/rustup \
    CARGO_HOME=/tmp/cargo \
    PATH=/tmp/cargo/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN yum -y update && \
    yum -y install gcc gcc-c++ make openssl-devel libffi-devel python3-devel git curl bzip2-devel zlib-devel && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain stable && \
    python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}" && \
    rm -rf /root/.cache/pip && \
    rm -rf /tmp/cargo /tmp/rustup

# Copy remaining project files
COPY . ${LAMBDA_TASK_ROOT}

# Default Lambda handler
CMD ["handler.lambda_handler"]
