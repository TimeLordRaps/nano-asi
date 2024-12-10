import os
import docker
from typing import Dict, List, Any, Optional

class DockerCodeExecutor:
    """
    Secure code execution environment using Docker.
    
    Provides isolated execution of code snippets with configurable constraints.
    """
    
    def __init__(
        self, 
        base_image: str = 'python:3.10-slim',
        memory_limit: str = '2g',
        cpu_limit: float = 1.0
    ):
        """
        Initialize Docker executor with resource constraints.
        
        Args:
            base_image: Base Docker image for execution
            memory_limit: Memory limit for container
            cpu_limit: CPU limit for container
        """
        self.client = docker.from_env()
        self.base_image = base_image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
    
    def execute_code(
        self, 
        code: str, 
        requirements: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute code in an isolated Docker container.
        
        Args:
            code: Python code to execute
            requirements: Optional list of pip requirements
        
        Returns:
            Execution results with stdout, stderr, and status
        """
        try:
            # Create temporary Dockerfile
            dockerfile_content = f"""
            FROM {self.base_image}
            WORKDIR /app
            {'RUN pip install ' + ' '.join(requirements) if requirements else ''}
            COPY code.py .
            CMD ["python", "code.py"]
            """
            
            # Write code and Dockerfile
            os.makedirs('/tmp/docker_exec', exist_ok=True)
            with open('/tmp/docker_exec/Dockerfile', 'w') as f:
                f.write(dockerfile_content)
            
            with open('/tmp/docker_exec/code.py', 'w') as f:
                f.write(code)
            
            # Build and run container
            container = self.client.containers.run(
                self.base_image,
                command=f"python /app/code.py",
                volumes={'/tmp/docker_exec': {'bind': '/app', 'mode': 'ro'}},
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=int(self.cpu_limit * 100000),
                remove=True,
                detach=True
            )
            
            # Wait for container and get results
            result = container.wait()
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            
            return {
                'status': result['StatusCode'],
                'stdout': stdout,
                'stderr': stderr
            }
        
        except Exception as e:
            return {
                'status': 1,
                'stdout': '',
                'stderr': str(e)
            }
        finally:
            # Clean up temporary files
            import shutil
            shutil.rmtree('/tmp/docker_exec', ignore_errors=True)
