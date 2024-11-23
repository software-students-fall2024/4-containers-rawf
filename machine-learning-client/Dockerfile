FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean

# Install pipenv
RUN pip install pipenv

# Copy Pipfile and Pipfile.lock first to leverage caching
COPY Pipfile Pipfile.lock ./

# Install dependencies using pipenv
RUN pipenv install --system --deploy

# Copy the rest of the application code
COPY . .

# Expose the necessary port
EXPOSE 5001

# Run the application
CMD ["python", "ml_client.py"]
