FROM python:3.9-slim-buster

# Update package lists
RUN apt-get update && apt-get install gcc g++ git build-essential -y

# Make working directories
RUN  mkdir -p  /open-gemini-deep-research
WORKDIR  /open-gemini-deep-research

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# Copy the .env file to the container
COPY .env .

# Copy every file in the source folder to the created working directory
COPY  . .

# Expose the port that the application will run on
EXPOSE 8080

# Start the application
CMD ["python3.9", "main.py"]