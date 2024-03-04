FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /usr/src/app

COPY aws_CV.py shuffled_train.csv ./

RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

RUN python3 -m venv venv

# Activate the virtual environment
# Install packages within the virtual environment
# Note: We use /usr/src/app/venv/bin/pip to ensure we use the pip from the virtual environment
RUN . venv/bin/activate && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install accelerate datasets evaluate transformers boto3 scikit-learn

# Ensure the command runs within the virtual environment
# Note: Use the Python interpreter from the virtual environment
CMD ["/usr/src/app/venv/bin/python3", "aws_CV.py"]
