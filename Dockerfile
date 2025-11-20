ARG PRIVATE_AWS_ECR_URL
FROM ${PRIVATE_AWS_ECR_URL}/pluangpython:3.9-v2

WORKDIR /workspace

# Configure runtime environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADEBOT_DATA_DIR=/workspace/data

    
RUN apt-get update
RUN apt-get install -y libpq-dev

# Copy requirements file
COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt


# Make startup scripts executable
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 8081

ENTRYPOINT ["/bin/bash"]

CMD ["start.sh"]
