#!/usr/bin/env python3
"""
Example demonstrating cloud deployment for distributed training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.cloud import CloudConfig, CloudDeployment, create_cloud_training_script


def create_sample_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create sample data for demonstration"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def create_model():
    """Create a sample model"""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )


def deploy_kubernetes():
    """Deploy to Kubernetes"""
    print("=== Kubernetes Deployment ===")
    
    # Create cloud configuration
    config = CloudConfig(
        platform="kubernetes",
        num_instances=4,
        instance_type="ml.p3.2xlarge",
        image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel",
        namespace="faeyon-training",
        storage_size="200Gi",
        node_selector={"node-type": "gpu"},
        resources={
            "requests": {"nvidia.com/gpu": 1, "memory": "16Gi", "cpu": "4"},
            "limits": {"nvidia.com/gpu": 1, "memory": "32Gi", "cpu": "8"}
        }
    )
    
    # Create deployment
    deployment = CloudDeployment(config)
    
    # Create training script
    training_script = """
# Your training code here
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
import torch.nn as nn

# Create model and optimizer
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
optimizer = FaeOptimizer("Adam", patterns=["*"], lr=0.001)

# Create recipe and train
recipe = ClassifyRecipe.create_distributed_recipe(model, optimizer)
# ... training code ...
"""
    
    # Deploy
    manifest_path = deployment.deploy(
        script_path="training.py",
        requirements=["torch", "torchvision", "numpy", "faeyon"]
    )
    
    print(f"Kubernetes manifest created: {manifest_path}")
    print("To deploy:")
    print(f"kubectl apply -f {manifest_path}")
    print("kubectl get pods -n faeyon-training")
    print("kubectl logs -f deployment/faeyon-training-<timestamp> -n faeyon-training")


def deploy_docker():
    """Deploy using Docker Compose"""
    print("\n=== Docker Compose Deployment ===")
    
    config = CloudConfig(
        platform="docker",
        num_instances=3,
        image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
    )
    
    deployment = CloudDeployment(config)
    
    compose_path = deployment.deploy(
        script_path="training.py",
        requirements=["torch", "torchvision", "numpy", "faeyon"]
    )
    
    print(f"Docker Compose file created: {compose_path}")
    print("To deploy:")
    print(f"docker-compose -f {compose_path} up")
    print("docker-compose -f {compose_path} logs -f")


def deploy_aws_sagemaker():
    """Deploy to AWS SageMaker"""
    print("\n=== AWS SageMaker Deployment ===")
    
    config = CloudConfig(
        platform="aws",
        region="us-west-2",
        instance_type="ml.p3.8xlarge",
        num_instances=2,
        image="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"
    )
    
    deployment = CloudDeployment(config)
    
    config_path = deployment.deploy(
        script_path="training.py",
        requirements=["torch", "torchvision", "numpy", "faeyon"]
    )
    
    print(f"AWS SageMaker config created: {config_path}")
    print("To deploy:")
    print("aws sagemaker create-training-job --cli-input-json file://<config_path>")
    print("aws sagemaker describe-training-job --training-job-name <job-name>")


def deploy_gcp_ai_platform():
    """Deploy to Google Cloud AI Platform"""
    print("\n=== Google Cloud AI Platform Deployment ===")
    
    config = CloudConfig(
        platform="gcp",
        region="us-central1",
        instance_type="n1-standard-4",
        num_instances=3,
        image="gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0:latest"
    )
    
    deployment = CloudDeployment(config)
    
    config_path = deployment.deploy(
        script_path="training.py",
        requirements=["torch", "torchvision", "numpy", "faeyon"]
    )
    
    print(f"GCP AI Platform config created: {config_path}")
    print("To deploy:")
    print("gcloud ai custom-jobs create --region=us-central1 --config=<config_path>")


def deploy_azure_ml():
    """Deploy to Azure ML"""
    print("\n=== Azure ML Deployment ===")
    
    config = CloudConfig(
        platform="azure",
        region="eastus",
        instance_type="Standard_NC6s_v3",
        num_instances=2,
        image="mcr.microsoft.com/azureml/pytorch-1.12-ubuntu20.04-py38-cuda11.6-gpu:latest"
    )
    
    deployment = CloudDeployment(config)
    
    config_path = deployment.deploy(
        script_path="training.py",
        requirements=["torch", "torchvision", "numpy", "faeyon"]
    )
    
    print(f"Azure ML config created: {config_path}")
    print("To deploy:")
    print("az ml job create --file <config_path>")


def create_training_script():
    """Create a complete training script for cloud deployment"""
    script_content = '''#!/usr/bin/env python3
"""
Cloud training script for distributed training.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import (
    ClassifyRecipe, FaeOptimizer, setup_distributed, cleanup_distributed
)

def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Starting distributed training with {world_size} processes")
        print(f"Using device: {device}")
    
    try:
        # Create sample data
        X = torch.randn(1000, 784)
        y = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Create model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
        # Create optimizer
        optimizer = FaeOptimizer("Adam", patterns=["*"], lr=0.001)
        
        # Create recipe
        recipe = ClassifyRecipe.create_distributed_recipe(model, optimizer)
        
        # Train
        history = recipe.train(
            dataloader, 
            "200s",  # Train for 200 steps
            val_data=dataloader,
            val_freq=50,
            verbose=(rank == 0)
        )
        
        if rank == 0:
            print(f"Training completed!")
            print(f"Total steps: {history['total_steps']}")
            print(f"Total time: {history['total_time']:.2f}s")
            print(f"Final loss: {history['train_loss'][-1]:.4f}")
    
    finally:
        # Cleanup
        cleanup_distributed()

if __name__ == "__main__":
    main()
'''
    
    with open("training.py", "w") as f:
        f.write(script_content)
    
    print("Training script created: training.py")


def main():
    """Main function demonstrating cloud deployment options"""
    print("Faeyon Cloud Deployment Examples")
    print("=" * 50)
    
    # Create training script
    create_training_script()
    
    # Demonstrate different cloud platforms
    deploy_kubernetes()
    deploy_docker()
    deploy_aws_sagemaker()
    deploy_gcp_ai_platform()
    deploy_azure_ml()
    
    print("\n" + "=" * 50)
    print("Cloud deployment examples completed!")
    print("\nNext steps:")
    print("1. Choose your preferred cloud platform")
    print("2. Update the configuration with your specific settings")
    print("3. Deploy using the provided commands")
    print("4. Monitor training progress using cloud platform tools")


if __name__ == "__main__":
    main()
