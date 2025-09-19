# Faeyon Cloud Deployment Guide

This guide demonstrates how to deploy Faeyon training jobs to various cloud platforms for distributed training.

## Supported Platforms

- **Kubernetes**: Deploy to any Kubernetes cluster (EKS, GKE, AKS, on-premises)
- **Docker Compose**: Local multi-GPU training with Docker
- **AWS SageMaker**: Managed training on AWS
- **Google Cloud AI Platform**: Managed training on GCP
- **Azure ML**: Managed training on Azure

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Training Script

```python
from faeyon.recipes import ClassifyRecipe, FaeOptimizer, setup_distributed, cleanup_distributed
import torch
import torch.nn as nn

def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Your training code here
    model = nn.Sequential(nn.Linear(784, 10))
    optimizer = FaeOptimizer("Adam", patterns=["*"], lr=0.001)
    recipe = ClassifyRecipe.create_distributed_recipe(model, optimizer)
    
    # Train
    history = recipe.train(train_loader, "1000s", verbose=(rank == 0))
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
```

### 3. Deploy to Cloud

```python
from faeyon.recipes import CloudConfig, CloudDeployment

# Configure for your platform
config = CloudConfig(
    platform="kubernetes",  # or "docker", "aws", "gcp", "azure"
    num_instances=4,
    instance_type="ml.p3.2xlarge"
)

# Deploy
deployment = CloudDeployment(config)
manifest_path = deployment.deploy("training.py")
```

## Platform-Specific Instructions

### Kubernetes

#### Prerequisites
- Kubernetes cluster with GPU support
- NVIDIA GPU operator installed
- kubectl configured

#### Deploy
```bash
# Generate manifest
python cloud_deployment_example.py

# Deploy to cluster
kubectl apply -f faeyon-training-<timestamp>-k8s.yaml

# Monitor training
kubectl get pods -n faeyon-training
kubectl logs -f deployment/faeyon-training-<timestamp> -n faeyon-training
```

#### Custom Configuration
```python
config = CloudConfig(
    platform="kubernetes",
    num_instances=8,
    instance_type="ml.p3.8xlarge",
    namespace="my-training",
    node_selector={"node-type": "gpu"},
    resources={
        "requests": {"nvidia.com/gpu": 1, "memory": "32Gi"},
        "limits": {"nvidia.com/gpu": 1, "memory": "64Gi"}
    }
)
```

### Docker Compose

#### Prerequisites
- Docker with GPU support
- NVIDIA Container Toolkit

#### Deploy
```bash
# Generate compose file
python cloud_deployment_example.py

# Start training
docker-compose -f faeyon-training-<timestamp>-docker-compose.yaml up

# Monitor logs
docker-compose -f faeyon-training-<timestamp>-docker-compose.yaml logs -f
```

### AWS SageMaker

#### Prerequisites
- AWS CLI configured
- SageMaker execution role
- S3 bucket for data and outputs

#### Deploy
```bash
# Generate SageMaker config
python cloud_deployment_example.py

# Create training job
aws sagemaker create-training-job --cli-input-json file://faeyon-training-<timestamp>-sagemaker.json

# Monitor training
aws sagemaker describe-training-job --training-job-name faeyon-training-<timestamp>
```

#### Custom Configuration
```python
config = CloudConfig(
    platform="aws",
    region="us-west-2",
    instance_type="ml.p3.16xlarge",
    num_instances=4
)
```

### Google Cloud AI Platform

#### Prerequisites
- Google Cloud SDK installed
- AI Platform API enabled
- Service account with proper permissions

#### Deploy
```bash
# Generate GCP config
python cloud_deployment_example.py

# Submit training job
gcloud ai custom-jobs create --region=us-central1 --config=faeyon-training-<timestamp>-gcp.json

# Monitor training
gcloud ai custom-jobs describe faeyon-training-<timestamp> --region=us-central1
```

### Azure ML

#### Prerequisites
- Azure CLI installed
- Azure ML workspace
- Compute cluster with GPU support

#### Deploy
```bash
# Generate Azure config
python cloud_deployment_example.py

# Submit training job
az ml job create --file faeyon-training-<timestamp>-azure.json

# Monitor training
az ml job show --name faeyon-training-<timestamp>
```

## Advanced Configuration

### Custom Docker Images

```python
config = CloudConfig(
    platform="kubernetes",
    image="your-registry/faeyon-training:latest",
    # ... other config
)
```

### Resource Requirements

```python
config = CloudConfig(
    platform="kubernetes",
    resources={
        "requests": {
            "nvidia.com/gpu": 2,
            "memory": "64Gi",
            "cpu": "8"
        },
        "limits": {
            "nvidia.com/gpu": 2,
            "memory": "128Gi",
            "cpu": "16"
        }
    }
)
```

### Node Selection

```python
config = CloudConfig(
    platform="kubernetes",
    node_selector={
        "node-type": "gpu",
        "instance-type": "p3.2xlarge"
    },
    tolerations=[{
        "key": "nvidia.com/gpu",
        "operator": "Exists",
        "effect": "NoSchedule"
    }]
)
```

## Monitoring and Debugging

### Kubernetes
```bash
# Check pod status
kubectl get pods -n faeyon-training

# View logs
kubectl logs -f <pod-name> -n faeyon-training

# Describe pod for debugging
kubectl describe pod <pod-name> -n faeyon-training

# Access pod shell
kubectl exec -it <pod-name> -n faeyon-training -- /bin/bash
```

### Docker Compose
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f worker-0

# Access container shell
docker-compose exec worker-0 /bin/bash
```

### Cloud Platforms
- **AWS**: Use SageMaker console or CloudWatch
- **GCP**: Use AI Platform console or Cloud Logging
- **Azure**: Use ML Studio or Application Insights

## Best Practices

### 1. Resource Optimization
- Use appropriate instance types for your workload
- Set proper resource requests and limits
- Monitor GPU utilization

### 2. Data Management
- Use persistent volumes for large datasets
- Implement data sharding for very large datasets
- Consider data preprocessing in the cloud

### 3. Cost Optimization
- Use spot instances when possible
- Implement auto-scaling
- Monitor and optimize training time

### 4. Security
- Use IAM roles and service accounts
- Encrypt data at rest and in transit
- Implement network security groups

### 5. Monitoring
- Set up logging and monitoring
- Use distributed tracing
- Implement alerting for failures

## Troubleshooting

### Common Issues

1. **GPU not detected**
   - Check NVIDIA drivers and container runtime
   - Verify GPU operator installation

2. **Out of memory**
   - Reduce batch size
   - Use gradient accumulation
   - Increase instance memory

3. **Network connectivity**
   - Check firewall rules
   - Verify DNS resolution
   - Test connectivity between nodes

4. **Permission errors**
   - Check IAM roles and policies
   - Verify service account permissions
   - Review security groups

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check network connectivity
ping <master-addr>

# Check distributed training setup
python -c "import torch; print(torch.distributed.is_available())"

# Test training script locally
python training.py
```

## Examples

See `cloud_deployment_example.py` for complete examples of deploying to each platform.

## Support

For issues and questions:
- Check the troubleshooting section
- Review platform-specific documentation
- Open an issue on GitHub
