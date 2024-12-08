# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy application code into the container
COPY . .

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the retinawarp-master directory into the container
COPY retinawarp-master /app/retinawarp-master

# Install the additional library manually using setup.py
RUN cd /app/retinawarp-master && pip install .

# Modify retina.py to remove 'pad' import after installation
RUN sed -i '/from skimage.util import crop, pad/s/, pad//' \
    /usr/local/lib/python3.9/site-packages/retina/retina.py

# Expose port 8080 to the outside world
EXPOSE 8080

# Run app.py when the container launches
CMD ["python3", "app.py"]