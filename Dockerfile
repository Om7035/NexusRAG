# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY pyproject.toml setup.cfg README.md ./
COPY nexusrag ./nexusrag

# Install dependencies
RUN pip install --no-cache-dir -e .

# Command to run when the container starts
CMD ["python", "app.py"]
