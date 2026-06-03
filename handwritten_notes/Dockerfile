FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for WeasyPrint
RUN yum update -y && \
    yum install -y \
    pango \
    pango-devel \
    cairo \
    cairo-devel \
    libffi-devel \
    python3-dev

WORKDIR /var/task

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy internal source (includes notebook.html)
COPY src/ /var/task/src/

# Copy main handler
COPY handler.py /var/task/

# Match the function name in handler.py
CMD ["handler.lambda_handler"]