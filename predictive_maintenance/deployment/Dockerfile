# Use a minimal base image with Python 3.10 installed
FROM python:3.10-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the project directory to /app
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create a non-root user for security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy app files with proper ownership
COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \  # changed from 8501 to 7860 to avoid conflict with hugging face
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
