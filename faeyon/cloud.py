"""
Cloud deployment utilities for Faeyon distributed training.

This module provides classes and functions for deploying Faeyon training jobs
to various cloud platforms including Kubernetes, Docker, AWS SageMaker,
Google Cloud AI Platform, and Azure ML.
"""

import json
import os
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

try:
    from google.cloud import aiplatform
except ImportError:
    aiplatform = None

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import AmlCompute, Environment, Job
except ImportError:
    MLClient = None


class CloudConfig:
    """Configuration for cloud-based distributed training"""
    
    def __init__(
        self,
        platform: str = "kubernetes",
        region: str = "us-west-2",
        instance_type: str = "ml.p3.2xlarge",
        num_instances: int = 2,
        image: str = "pytorch/pytorch:latest",
        namespace: str = "default",
        storage_class: str = "gp2",
        storage_size: str = "100Gi",
        node_selector: Optional[Dict[str, str]] = None,
        tolerations: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[Dict[str, Any]] = None
    ):
        self.platform = platform
        self.region = region
        self.instance_type = instance_type
        self.num_instances = num_instances
        self.image = image
        self.namespace = namespace
        self.storage_class = storage_class
        self.storage_size = storage_size
        self.node_selector = node_selector or {}
        self.tolerations = tolerations or []
        self.resources = resources or {
            "requests": {"nvidia.com/gpu": 1},
            "limits": {"nvidia.com/gpu": 1}
        }


class CloudDeployment:
    """Handles cloud deployment for distributed training"""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.deployment_name = f"faeyon-training-{int(time.time())}"
    
    def create_kubernetes_manifest(self, script_path: str, requirements: Optional[List[str]] = None) -> str:
        """Create Kubernetes manifest for distributed training"""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_name,
                "namespace": self.config.namespace,
                "labels": {"app": "faeyon-training"}
            },
            "spec": {
                "replicas": self.config.num_instances,
                "selector": {"matchLabels": {"app": "faeyon-training"}},
                "template": {
                    "metadata": {"labels": {"app": "faeyon-training"}},
                    "spec": {
                        "containers": [{
                            "name": "faeyon-training",
                            "image": self.config.image,
                            "command": ["torchrun"],
                            "args": [
                                "--nproc_per_node=1",
                                "--nnodes={}".format(self.config.num_instances),
                                "--node_rank=$(NODE_RANK)",
                                "--master_addr=$(MASTER_ADDR)",
                                "--master_port=$(MASTER_PORT)",
                                script_path
                            ],
                            "env": [
                                {"name": "NODE_RANK", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                                {"name": "MASTER_ADDR", "value": f"{self.deployment_name}-0"},
                                {"name": "MASTER_PORT", "value": "29500"}
                            ],
                            "resources": self.config.resources,
                            "volumeMounts": [{
                                "name": "training-data",
                                "mountPath": "/data"
                            }]
                        }],
                        "volumes": [{
                            "name": "training-data",
                            "persistentVolumeClaim": {
                                "claimName": f"{self.deployment_name}-pvc"
                            }
                        }],
                        "nodeSelector": self.config.node_selector,
                        "tolerations": self.config.tolerations
                    }
                }
            }
        }
        
        # Add PersistentVolumeClaim
        pvc_manifest = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.deployment_name}-pvc",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteMany"],
                "resources": {
                    "requests": {"storage": self.config.storage_size}
                },
                "storageClassName": self.config.storage_class
            }
        }
        
        if yaml is None:
            raise ImportError("PyYAML is required for Kubernetes manifest generation. Install with: pip install pyyaml")
        return yaml.dump_all([manifest, pvc_manifest], default_flow_style=False)
    
    def create_docker_compose(self, script_path: str, requirements: Optional[List[str]] = None) -> str:
        """Create Docker Compose file for distributed training"""
        compose: Dict[str, Any] = {
            "version": "3.8",
            "services": {}
        }
        
        for i in range(self.config.num_instances):
            service_name = f"worker-{i}"
            compose["services"][service_name] = {
                "image": self.config.image,
                "command": [
                    "torchrun",
                    "--nproc_per_node=1",
                    f"--nnodes={self.config.num_instances}",
                    f"--node_rank={i}",
                    "--master_addr=worker-0",
                    "--master_port=29500",
                    script_path
                ],
                "environment": [
                    f"NODE_RANK={i}",
                    "MASTER_ADDR=worker-0",
                    "MASTER_PORT=29500"
                ],
                "volumes": ["./data:/data", f"./{script_path}:/app/{script_path}"],
                "deploy": {
                    "resources": {
                        "reservations": {"devices": [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]}
                    }
                }
            }
        
        if yaml is None:
            raise ImportError("PyYAML is required for Docker Compose generation. Install with: pip install pyyaml")
        return yaml.dump(compose, default_flow_style=False)
    
    def create_aws_sagemaker_config(self, script_path: str, requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create AWS SageMaker configuration for distributed training"""
        return {
            "TrainingJobName": self.deployment_name,
            "RoleArn": "arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole",
            "AlgorithmSpecification": {
                "TrainingInputMode": "File",
                "TrainingImage": self.config.image
            },
            "InputDataConfig": [{
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": "s3://your-bucket/training-data/",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                }
            }],
            "OutputDataConfig": {
                "S3OutputPath": "s3://your-bucket/output/"
            },
            "ResourceConfig": {
                "InstanceType": self.config.instance_type,
                "InstanceCount": self.config.num_instances,
                "VolumeSizeInGB": 100
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 3600
            },
            "HyperParameters": {
                "sagemaker_program": script_path,
                "sagemaker_submit_directory": "/opt/ml/code",
                "sagemaker_region": self.config.region
            }
        }
    
    def create_gcp_ai_platform_config(self, script_path: str, requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create Google Cloud AI Platform configuration for distributed training"""
        return {
            "displayName": self.deployment_name,
            "trainingInput": {
                "scaleTier": "CUSTOM",
                "masterType": self.config.instance_type,
                "workerType": self.config.instance_type,
                "workerCount": self.config.num_instances - 1,
                "region": self.config.region,
                "pythonModule": "trainer.task",
                "packageUris": ["gs://your-bucket/packages/trainer-1.0.tar.gz"],
                "args": [
                    f"--nproc_per_node=1",
                    f"--nnodes={self.config.num_instances}",
                    "--master_addr=$(MASTER_ADDR)",
                    "--master_port=$(MASTER_PORT)",
                    script_path
                ],
                "masterConfig": {
                    "imageUri": self.config.image
                },
                "workerConfig": {
                    "imageUri": self.config.image
                }
            }
        }
    
    def create_azure_ml_config(self, script_path: str, requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create Azure ML configuration for distributed training"""
        return {
            "name": self.deployment_name,
            "type": "Microsoft.MachineLearningServices/workspaces/computes",
            "properties": {
                "computeType": "AmlCompute",
                "properties": {
                    "vmSize": self.config.instance_type,
                    "vmPriority": "Dedicated",
                    "scaleSettings": {
                        "maxNodeCount": self.config.num_instances,
                        "minNodeCount": 0,
                        "nodeIdleTimeBeforeScaleDown": "PT5M"
                    }
                }
            },
            "script": {
                "source_directory": ".",
                "script": script_path,
                "environment": {
                    "name": "pytorch-env",
                    "docker": {"image": self.config.image},
                    "python": {
                        "conda_dependencies": {
                            "channels": ["pytorch", "conda-forge"],
                            "dependencies": requirements or ["pytorch", "torchvision", "numpy"]
                        }
                    }
                }
            }
        }
    
    def deploy(self, script_path: str, requirements: Optional[List[str]] = None, output_dir: str = ".", 
               create_infrastructure: bool = True, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Deploy distributed training configuration and optionally create infrastructure.
        
        Args:
            script_path: Path to training script
            requirements: Optional requirements list
            output_dir: Directory to save deployment files
            create_infrastructure: Whether to actually create cloud resources
            **kwargs: Additional platform-specific parameters
            
        Returns:
            Path to generated config file or deployment result dict
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.config.platform == "kubernetes":
            manifest = self.create_kubernetes_manifest(script_path, requirements)
            manifest_path = output_path / f"{self.deployment_name}-k8s.yaml"
            with open(manifest_path, 'w') as f:
                f.write(manifest)
            
            if create_infrastructure:
                return self._deploy_kubernetes(str(manifest_path), **kwargs)
            return str(manifest_path)
        
        elif self.config.platform == "docker":
            compose = self.create_docker_compose(script_path, requirements)
            compose_path = output_path / f"{self.deployment_name}-docker-compose.yaml"
            with open(compose_path, 'w') as f:
                f.write(compose)
            
            if create_infrastructure:
                return self._deploy_docker(str(compose_path), **kwargs)
            return str(compose_path)
        
        elif self.config.platform == "aws":
            config = self.create_aws_sagemaker_config(script_path, requirements)
            config_path = output_path / f"{self.deployment_name}-sagemaker.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if create_infrastructure:
                return self._deploy_aws(config, **kwargs)
            return str(config_path)
        
        elif self.config.platform == "gcp":
            config = self.create_gcp_ai_platform_config(script_path, requirements)
            config_path = output_path / f"{self.deployment_name}-gcp.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if create_infrastructure:
                return self._deploy_gcp(config, **kwargs)
            return str(config_path)
        
        elif self.config.platform == "azure":
            config = self.create_azure_ml_config(script_path, requirements)
            config_path = output_path / f"{self.deployment_name}-azure.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            if create_infrastructure:
                return self._deploy_azure(config, **kwargs)
            return str(config_path)
        
        else:
            raise ValueError(f"Unsupported platform: {self.config.platform}")
    
    def _deploy_kubernetes(self, manifest_path: str, **kwargs) -> Dict[str, Any]:
        """Deploy to Kubernetes cluster"""
        try:
            # Apply the manifest
            result = subprocess.run(
                ["kubectl", "apply", "-f", manifest_path],
                capture_output=True, text=True, check=True
            )
            
            # Wait for deployment to be ready
            subprocess.run(
                ["kubectl", "wait", "--for=condition=available", 
                 f"deployment/{self.deployment_name}", f"--timeout={kwargs.get('timeout', '300s')}"],
                capture_output=True, text=True, check=True
            )
            
            return {
                "status": "success",
                "platform": "kubernetes",
                "deployment_name": self.deployment_name,
                "manifest_path": manifest_path,
                "output": result.stdout,
                "logs": f"kubectl logs deployment/{self.deployment_name}"
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "platform": "kubernetes",
                "error": str(e),
                "output": e.stdout,
                "stderr": e.stderr
            }
    
    def _deploy_docker(self, compose_path: str, **kwargs) -> Dict[str, Any]:
        """Deploy using Docker Compose"""
        try:
            # Start the services
            result = subprocess.run(
                ["docker-compose", "-f", compose_path, "up", "-d"],
                capture_output=True, text=True, check=True
            )
            
            return {
                "status": "success",
                "platform": "docker",
                "deployment_name": self.deployment_name,
                "compose_path": compose_path,
                "output": result.stdout,
                "logs": f"docker-compose -f {compose_path} logs"
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error",
                "platform": "docker",
                "error": str(e),
                "output": e.stdout,
                "stderr": e.stderr
            }
    
    def _deploy_aws(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Deploy to AWS SageMaker"""
        if boto3 is None:
            return {
                "status": "error",
                "platform": "aws",
                "error": "boto3 not installed. Install with: pip install boto3"
            }
        
        try:
            sagemaker = boto3.client('sagemaker', region_name=self.config.region)
            
            # Create training job
            response = sagemaker.create_training_job(**config)
            
            return {
                "status": "success",
                "platform": "aws",
                "deployment_name": self.deployment_name,
                "training_job_name": response['TrainingJobArn'],
                "monitor": f"aws sagemaker describe-training-job --training-job-name {self.deployment_name}"
            }
        except ClientError as e:
            return {
                "status": "error",
                "platform": "aws",
                "error": str(e),
                "error_code": e.response['Error']['Code']
            }
    
    def _deploy_gcp(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Deploy to Google Cloud AI Platform"""
        if aiplatform is None:
            return {
                "status": "error",
                "platform": "gcp",
                "error": "google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform"
            }
        
        try:
            # Initialize AI Platform
            aiplatform.init(project=kwargs.get('project_id'), location=self.config.region)
            
            # Create custom job
            job = aiplatform.CustomJob(
                display_name=self.deployment_name,
                worker_pool_specs=config['trainingInput']['workerPoolSpecs'],
                staging_bucket=kwargs.get('staging_bucket')
            )
            
            # Submit job
            job.submit()
            
            return {
                "status": "success",
                "platform": "gcp",
                "deployment_name": self.deployment_name,
                "job_name": job.resource_name,
                "monitor": f"gcloud ai custom-jobs describe {job.resource_name}"
            }
        except Exception as e:
            return {
                "status": "error",
                "platform": "gcp",
                "error": str(e)
            }
    
    def _deploy_azure(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Deploy to Azure ML"""
        if MLClient is None:
            return {
                "status": "error",
                "platform": "azure",
                "error": "azure-ai-ml not installed. Install with: pip install azure-ai-ml"
            }
        
        try:
            # Initialize ML Client
            ml_client = MLClient(
                credential=kwargs.get('credential'),
                subscription_id=kwargs.get('subscription_id'),
                resource_group_name=kwargs.get('resource_group_name'),
                workspace_name=kwargs.get('workspace_name')
            )
            
            # Create compute cluster if it doesn't exist
            compute_name = f"{self.deployment_name}-compute"
            try:
                compute = ml_client.compute.get(compute_name)
            except:
                compute = AmlCompute(
                    name=compute_name,
                    size=self.config.instance_type,
                    min_instances=0,
                    max_instances=self.config.num_instances
                )
                ml_client.compute.begin_create_or_update(compute)
            
            # Create environment
            env = Environment(
                name=f"{self.deployment_name}-env",
                image=self.config.image,
                conda_file=kwargs.get('conda_file')
            )
            ml_client.environments.create_or_update(env)
            
            # Create and submit job
            job = Job(
                display_name=self.deployment_name,
                compute=compute_name,
                environment=env.name,
                code=kwargs.get('code_path', '.'),
                command=config['script']['script']
            )
            
            ml_client.jobs.create_or_update(job)
            
            return {
                "status": "success",
                "platform": "azure",
                "deployment_name": self.deployment_name,
                "job_name": job.name,
                "monitor": f"az ml job show --name {job.name} --resource-group {kwargs.get('resource_group_name')}"
            }
        except Exception as e:
            return {
                "status": "error",
                "platform": "azure",
                "error": str(e)
            }
    
    def get_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of deployed job"""
        if self.config.platform == "kubernetes":
            try:
                result = subprocess.run(
                    ["kubectl", "get", "deployment", self.deployment_name, "-o", "json"],
                    capture_output=True, text=True, check=True
                )
                return json.loads(result.stdout)
            except subprocess.CalledProcessError as e:
                return {"status": "error", "error": str(e)}
        
        elif self.config.platform == "docker":
            try:
                result = subprocess.run(
                    ["docker-compose", "ps", "--format", "json"],
                    capture_output=True, text=True, check=True
                )
                return {"status": "success", "containers": result.stdout}
            except subprocess.CalledProcessError as e:
                return {"status": "error", "error": str(e)}
        
        elif self.config.platform == "aws" and boto3:
            try:
                sagemaker = boto3.client('sagemaker', region_name=self.config.region)
                response = sagemaker.describe_training_job(
                    TrainingJobName=job_id or self.deployment_name
                )
                return {
                    "status": "success",
                    "training_job_status": response['TrainingJobStatus'],
                    "creation_time": response['CreationTime'],
                    "last_modified_time": response['LastModifiedTime']
                }
            except ClientError as e:
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_implemented", "platform": self.config.platform}
    
    def stop_job(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop running job"""
        if self.config.platform == "kubernetes":
            try:
                result = subprocess.run(
                    ["kubectl", "delete", "deployment", self.deployment_name],
                    capture_output=True, text=True, check=True
                )
                return {"status": "success", "output": result.stdout}
            except subprocess.CalledProcessError as e:
                return {"status": "error", "error": str(e)}
        
        elif self.config.platform == "docker":
            try:
                result = subprocess.run(
                    ["docker-compose", "down"],
                    capture_output=True, text=True, check=True
                )
                return {"status": "success", "output": result.stdout}
            except subprocess.CalledProcessError as e:
                return {"status": "error", "error": str(e)}
        
        elif self.config.platform == "aws" and boto3:
            try:
                sagemaker = boto3.client('sagemaker', region_name=self.config.region)
                sagemaker.stop_training_job(
                    TrainingJobName=job_id or self.deployment_name
                )
                return {"status": "success", "message": "Training job stopped"}
            except ClientError as e:
                return {"status": "error", "error": str(e)}
        
        return {"status": "not_implemented", "platform": self.config.platform}


def create_cloud_training_script(
    training_script: str,
    output_path: str = "cloud_training.py"
) -> str:
    """Create a cloud-optimized training script"""
    
    script_content = f'''#!/usr/bin/env python3
"""
Cloud-optimized training script for distributed training.
Generated by Faeyon CloudDeployment.
"""

import os
import sys
import torch
import torch.distributed as dist
from faeyon.recipes import setup_distributed, cleanup_distributed

def main():
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device
    device = torch.device(f'cuda:{{local_rank}}' if torch.cuda.is_available() else 'cpu')
    
    # Import your training code here
    # {training_script}
    
    try:
        # Your training code goes here
        print(f"Starting training on rank {{rank}}/{{world_size}} on device {{device}}")
        
        # Example training code (replace with your actual training)
        from faeyon.recipes import ClassifyRecipe, FaeOptimizer
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data for example
        X = torch.randn(1000, 784)
        y = torch.randint(0, 10, (1000,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Create model
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        # Create recipe
        optimizer = FaeOptimizer("Adam", patterns=["*"], lr=0.001)
        recipe = ClassifyRecipe.create_distributed_recipe(model, optimizer)
        
        # Train
        history = recipe.train(dataloader, "100s", verbose=(rank == 0))
        
        if rank == 0:
            print(f"Training completed: {{history['total_steps']}} steps in {{history['total_time']:.2f}}s")
    
    finally:
        # Cleanup
        cleanup_distributed()

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_path, 0o755)
    
    return output_path


def deploy_training_job(
    script_path: str,
    platform: str = "kubernetes",
    create_infrastructure: bool = True,
    **config_kwargs
) -> Union[str, Dict[str, Any]]:
    """
    Convenience function to deploy a training job to the cloud.
    
    Args:
        script_path: Path to training script
        platform: Cloud platform ("kubernetes", "docker", "aws", "gcp", "azure")
        create_infrastructure: Whether to actually create cloud resources
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Deployment result or config file path
    """
    config = CloudConfig(platform=platform, **config_kwargs)
    deployment = CloudDeployment(config)
    
    return deployment.deploy(
        script_path=script_path,
        create_infrastructure=create_infrastructure,
        **config_kwargs
    )


def list_cloud_jobs(platform: str, **kwargs) -> List[Dict[str, Any]]:
    """
    List running cloud training jobs.
    
    Args:
        platform: Cloud platform
        **kwargs: Platform-specific parameters
        
    Returns:
        List of job information
    """
    if platform == "kubernetes":
        try:
            result = subprocess.run(
                ["kubectl", "get", "deployments", "-l", "app=faeyon-training", "-o", "json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)
            return [item for item in data.get('items', [])]
        except subprocess.CalledProcessError:
            return []
    
    elif platform == "docker":
        try:
            result = subprocess.run(
                ["docker-compose", "ps", "--format", "json"],
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout) if result.stdout.strip() else []
        except subprocess.CalledProcessError:
            return []
    
    elif platform == "aws" and boto3:
        try:
            sagemaker = boto3.client('sagemaker', region_name=kwargs.get('region', 'us-west-2'))
            response = sagemaker.list_training_jobs(
                NameContains='faeyon-training',
                MaxResults=100
            )
            return response.get('TrainingJobSummaries', [])
        except ClientError:
            return []
    
    return []
