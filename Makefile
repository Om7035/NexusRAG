.PHONY: install test clean docs serve-docs run-app create-test-pdf

# Install the package in development mode
install:
	pip install -e .

# Run all tests
test:
	pytest

# Run tests with coverage
test-cov:
	pytest --cov=nexusrag --cov-report=html --cov-report=term

# Clean build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Build documentation
docs:
	mkdocs build

# Serve documentation locally
serve-docs:
	mkdocs serve

# Run the Streamlit demo application
run-app:
	streamlit run app.py

# Create a test PDF
create-test-pdf:
	python scripts/create_test_pdf.py

# Run integration test
integration-test:
	python scripts/integration_test.py
