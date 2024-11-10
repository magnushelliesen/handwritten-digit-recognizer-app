# Base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port Cloud Run uses
EXPOSE 8080

# Command to run the app
CMD ["streamlit", "run", "0_🤖_Handwritten_digit_recognizer.py", "--server.port=8080", "--server.enableCORS=false", "--browser.serverAddress=0.0.0.0"]
