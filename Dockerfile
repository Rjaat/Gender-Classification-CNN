FROM streamlit/streamlit:latest

# Install OpenGL libraries
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# Set the working directory
WORKDIR /app


RUN apt-get update
RUN apt install -y libgl1-mesa-glx
# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]

