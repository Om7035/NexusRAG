import pytest
import sys
from unittest.mock import patch, MagicMock
import os
import tempfile
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_cli_version():
    """Test the CLI version command."""
    from nexusrag.cli import main
    
    with patch('sys.argv', ['nexusrag', 'version']):
        with patch('builtins.print') as mock_print:
            try:
                main()
            except SystemExit:
                pass
            
            # Check that print was called with version info
            mock_print.assert_called()


def test_cli_help():
    """Test the CLI help output."""
    from nexusrag.cli import main
    import argparse
    
    with patch('sys.argv', ['nexusrag']):
        with patch('argparse.ArgumentParser.print_help') as mock_help:
            try:
                main()
            except SystemExit:
                pass
            
            # Check that help was printed
            mock_help.assert_called_once()
