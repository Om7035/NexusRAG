# Installation

This guide will help you install NexusRAG on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Options

### Option 1: Install from PyPI (Recommended when available)

```bash
pip install nexusrag
```

### Option 2: Install from Source

1. Clone the repository:

```bash
git clone https://github.com/your-username/NexusRAG.git
cd NexusRAG
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:

```bash
pip install -e .
```

### Option 3: Install Dependencies Manually

1. Clone the repository:

```bash
git clone https://github.com/your-username/NexusRAG.git
cd NexusRAG
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Verify Installation

To verify that NexusRAG is installed correctly, you can run:

```python
import nexusrag
print(nexusrag.__version__)
```

## Running Tests

To run the test suite, use:

```bash
pytest
```

## Running the Demo Application

To run the Streamlit demo application:

```bash
streamlit run app.py
```

## Troubleshooting

If you encounter any issues during installation, please check the following:

1. Ensure you have Python 3.8 or higher installed
2. Ensure pip is up to date: `pip install --upgrade pip`
3. If you're having issues with dependencies, try installing them one by one
4. Check the [GitHub issues](https://github.com/your-username/NexusRAG/issues) for similar problems

If you're still having trouble, please open an issue on GitHub.
