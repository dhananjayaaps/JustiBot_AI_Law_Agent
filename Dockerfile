# Use an official Python runtime as the base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install project dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the port FastAPI is running on
EXPOSE 8000

# Define the command to start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
