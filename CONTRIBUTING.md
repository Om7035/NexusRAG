## Development Environment Setup

### Prerequisites
- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- Docker Desktop (for container-based development)

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Om7035/NexusRAG.git
   cd NexusRAG
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```

4. For PDF processing support, install system dependencies:
   ```bash
   # Ubuntu
   sudo apt-get install poppler-utils tesseract-ocr
   
   # Windows (requires Chocolatey)
   choco install poppler tesseract
   ```

5. Run tests:
   ```bash
   pytest tests/
   ```

### Docker-based Development
```bash
docker compose up --build -d
docker compose logs -f  # Monitor logs
