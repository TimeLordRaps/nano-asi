#!/bin/bash

# Create wiki directory structure
mkdir -p wiki/Getting_Started
mkdir -p wiki/Core_Architecture
mkdir -p wiki/Development
mkdir -p wiki/Tutorials
mkdir -p wiki/Components
mkdir -p wiki/Advanced
mkdir -p wiki/API_Reference
mkdir -p wiki/Research

# Move and consolidate documentation files
cp docs/README.md wiki/README.md
cp docs/philosophy.md wiki/Philosophy.md
cp docs/core_concepts.md wiki/Core_Architecture/Overview.md
cp docs/development/contributing.md wiki/Development/Contributing.md
cp docs/development/testing.md wiki/Development/Testing.md
cp docs/tutorials/basic_usage.md wiki/Tutorials/Basic_Usage.md
cp docs/tutorials/advanced_configuration.md wiki/Tutorials/Advanced_Configuration.md

# Remove old docs directory
rm -rf docs

echo "Documentation migration complete!"
