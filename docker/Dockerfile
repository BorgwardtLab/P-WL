# Use prebuild python environment
FROM python:3

# Set working directory
WORKDIR /ICML2019

# Copy requirements.txt
COPY . /ICML2019

# Install pipenv 
RUN pip install --trusted-host pypi.org -r requirements.txt
