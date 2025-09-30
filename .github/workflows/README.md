# GitHub Actions Workflows

This directory contains CI/CD workflows for NexusRAG:

## Workflows

1. **ci.yml** - Continuous Integration
   - Runs on: push, pull_request
   - Jobs:
     - Linting (flake8)
     - Unit testing (pytest)
     - Integration testing
     - Security scanning (Trivy)

2. **docker-publish.yml** - Docker Image Publishing
   - Runs on: release
   - Jobs:
     - Build Docker images
     - Push to Docker Hub/GitHub Container Registry

3. **deploy.yml** - Deployment
   - Runs on: main branch updates
   - Jobs:
     - Deploy to staging environment
     - Run smoke tests
     - Approve production deployment
     - Deploy to production

## Usage

Workflows run automatically based on GitHub events. To manually trigger:
```bash
gh workflow run [workflow-name]
```
