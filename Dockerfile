FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Update package lists and install necessary tools
RUN apt-get update && apt-get install -y \
    dnsutils \
    iputils-ping \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    net-tools \
    traceroute \
    telnet \
    tcpdump \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Create a DNS configuration directory for our application
RUN mkdir -p /app/dns_config

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and startup script
COPY ./app ./app
COPY startup.sh /app/startup.sh

# Make the startup script executable
RUN chmod +x /app/startup.sh

# Expose the port the app runs on
EXPOSE 8000

# Use the startup script to run pre-checks and then start the application
ENTRYPOINT ["/app/startup.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]