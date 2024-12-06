"""Colab setup script for nano-asi."""

import subprocess
import sys
import os

def setup_colab():
    """Set up nano-asi in Google Colab environment."""
    print("Setting up nano-asi in Colab...")
    
    # Clone repository from TimeLordRaps
    subprocess.run(["git", "clone", "https://github.com/TimeLordRaps/nano-asi.git"], check=True)
    
    # Install dependencies
    subprocess.run(["pip", "install", "-r", "nano-asi/requirements.txt"], check=True)
    subprocess.run(["pip", "install", "-e", "nano-asi"], check=True)
    
    # Add to Python path
    sys.path.append('/content/nano-asi')
    
    # Install additional dependencies
    subprocess.run([
        "pip", "install", 
        "torch", "transformers", "datasets", "accelerate", 
        "bitsandbytes", "nest-asyncio"
    ], check=True)
    
    print("Setup complete! You can now import and use nano-asi.")

def run_example():
    """Run a simple example using nano-asi."""
    import nest_asyncio
    nest_asyncio.apply()
    
    from nano_asi import ASI
    import asyncio
    
    async def main():
        asi = ASI()
        result = await asi.run("Generate an innovative solution for climate change")
        print("\nGenerated Solution:")
        print("-" * 40)
        print(result.solution)
        print("-" * 40)
        print("\nConsciousness Flow Metrics:")
        print(result.metrics)
    
    # Run the async function
    asyncio.run(main())

if __name__ == "__main__":
    setup_colab()
    run_example()
