ARG PRIVATE_AWS_ECR_URL
FROM ${PRIVATE_AWS_ECR_URL}/pluangpython:3.9-v2
# Create a workspace directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . .

ENTRYPOINT ["python3"]
CMD ["mainEmptyRun.py"]
