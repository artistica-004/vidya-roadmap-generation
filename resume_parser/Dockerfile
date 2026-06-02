# Use AWS Lambda Python 3.10 base image
FROM public.ecr.aws/lambda/python:3.10

# Install minimal system deps
RUN yum -y install gcc gcc-c++ make libffi-devel git curl && \
    yum clean all

# Copy requirements first (better caching)
COPY requirements.txt .

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Install Python deps into Lambda task root
RUN python3.10 -m pip install \
        --no-cache-dir \
        --force-reinstall \
        --prefer-binary \
        -r requirements.txt \
        --target "${LAMBDA_TASK_ROOT}"

# Copy project files
COPY . ${LAMBDA_TASK_ROOT}

# Set handler
CMD ["handler.lambda_handler"]
