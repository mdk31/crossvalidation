FROM python:3.9
WORKDIR /usr/src/app
COPY train_dat.csv aws_CV.py requirements_aws.txt .
RUN pip install -r requirements_aws.txt
CMD ["python", "aws_CV.py"]