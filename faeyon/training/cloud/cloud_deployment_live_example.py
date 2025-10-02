#!/usr/bin/env python3
"""
Live example demonstrating actual cloud deployment and job management.
This example shows how to deploy training jobs to the cloud and manage them.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.cloud import CloudConfig, CloudDeployment, deploy_training_job, list_cloud_jobs


def create_sample_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create sample data for demonstration"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def create_training_script():
    """Create a simple training script for cloud deployment"""
    script_content = '''#!/usr/bin/env python3
"""
Cloud training script for Faeyon distributed training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer

def main():
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
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create recipe
    recipe = ClassifyRecipe.create_distributed_recipe(model, optimizer)
    
    # Train
    history = recipe.train(dataloader, "50s", "100s", verbose=True)
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")

if __name__ == "__main__":
    main()
'''
    
    with open("cloud_training_script.py", "w") as f:
        f.write(script_content)
    
    return "cloud_training_script.py"


def main():
    print("=== Live Cloud Deployment Example ===")
    
    # Create training script
    script_path = create_training_script()
    print(f"Created training script: {script_path}")
    
    # Example 1: Deploy to Kubernetes (config only)
    print("\n--- 1. Generate Kubernetes Config (No Infrastructure) ---")
    config = CloudConfig(
        platform="kubernetes",
        num_instances=2,
        image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
    )
    deployment = CloudDeployment(config)
    
    # Generate config without creating infrastructure
    manifest_path = deployment.deploy(
        script_path=script_path,
        create_infrastructure=False,
        output_dir="k8s_config"
    )
    print(f"Kubernetes manifest generated: {manifest_path}")
    print("To deploy: kubectl apply -f " + manifest_path)
    
    # Example 2: Deploy to Docker (config only)
    print("\n--- 2. Generate Docker Compose Config (No Infrastructure) ---")
    docker_config = CloudConfig(
        platform="docker",
        num_instances=2,
        image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel"
    )
    docker_deployment = CloudDeployment(docker_config)
    
    compose_path = docker_deployment.deploy(
        script_path=script_path,
        create_infrastructure=False,
        output_dir="docker_config"
    )
    print(f"Docker Compose file generated: {compose_path}")
    print("To run: docker-compose -f " + compose_path + " up")
    
    # Example 3: Deploy to AWS SageMaker (config only)
    print("\n--- 3. Generate AWS SageMaker Config (No Infrastructure) ---")
    aws_config = CloudConfig(
        platform="aws",
        region="us-west-2",
        instance_type="ml.p3.2xlarge",
        num_instances=1
    )
    aws_deployment = CloudDeployment(aws_config)
    
    sagemaker_path = aws_deployment.deploy(
        script_path=script_path,
        create_infrastructure=False,
        output_dir="aws_config"
    )
    print(f"AWS SageMaker config generated: {sagemaker_path}")
    print("Remember to update 'RoleArn' and S3 paths before deploying")
    
    # Example 4: Live deployment (if you have the infrastructure)
    print("\n--- 4. Live Deployment Example (Requires Infrastructure) ---")
    print("This would actually deploy to the cloud if you have the infrastructure set up.")
    print("Uncomment the following lines to try live deployment:")
    
    # Uncomment these lines to try live deployment
    # try:
    #     # Deploy to Kubernetes (requires kubectl and cluster access)
    #     result = deploy_training_job(
    #         script_path=script_path,
    #         platform="kubernetes",
    #         num_instances=2,
    #         create_infrastructure=True
    #     )
    #     print(f"Kubernetes deployment result: {result}")
    #     
    #     # Check status
    #     status = deployment.get_status()
    #     print(f"Deployment status: {status}")
    #     
    # except Exception as e:
    #     print(f"Live deployment failed (expected if no infrastructure): {e}")
    
    # Example 5: list existing jobs
    print("\n--- 5. list Existing Cloud Jobs ---")
    for platform in ["kubernetes", "docker", "aws"]:
        jobs = list_cloud_jobs(platform)
        print(f"{platform.capitalize()} jobs: {len(jobs)}")
        for job in jobs[:3]:  # Show first 3 jobs
            if isinstance(job, dict):
                name = job.get('metadata', {}).get('name', job.get('TrainingJobName', 'Unknown'))
                print(f"  - {name}")
    
    # Example 6: Job management
    print("\n--- 6. Job Management Example ---")
    print("To manage jobs after deployment:")
    print("  - Check status: deployment.get_status()")
    print("  - Stop job: deployment.stop_job()")
    print("  - list jobs: list_cloud_jobs(platform)")
    
    print("\n=== Cloud Deployment Complete ===")
    print("Generated configuration files in:")
    print("  - k8s_config/ (Kubernetes)")
    print("  - docker_config/ (Docker Compose)")
    print("  - aws_config/ (AWS SageMaker)")
    print("\nTo deploy live, ensure you have:")
    print("  - kubectl configured for Kubernetes")
    print("  - Docker and docker-compose for Docker")
    print("  - AWS CLI configured for SageMaker")
    print("  - GCP SDK configured for AI Platform")
    print("  - Azure CLI configured for ML")


if __name__ == "__main__":
    main()
