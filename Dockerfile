# Use AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Install only minimal deps - NO git, NO openssl-devel (triggers snapsafe conflict)
RUN yum -y install gcc gcc-c++ make libffi-devel tar && \
    yum clean all

# Install rustup and add cargo to PATH
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --no-modify-path --default-toolchain stable && \
    rustc --version

# Copy requirements first (better layer caching)
COPY requirements.txt .

# Upgrade pip and install Python deps into Lambda task root
RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir \
        --prefer-binary \
        -r requirements.txt \
        --target "${LAMBDA_TASK_ROOT}"

# Copy project files into image
COPY . ${LAMBDA_TASK_ROOT}

# Handler
CMD ["handler.lambda_handler"]
