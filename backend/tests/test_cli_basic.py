"""
Basic CLI tests that don't require heavy ML dependencies.
"""

import pytest
from honesty.cli.main import create_parser, main


def test_cli_parser_creation():
    """Test that CLI parser can be created successfully."""
    parser = create_parser()
    assert parser is not None
    assert parser.prog == "honesty"


def test_cli_help():
    """Test that CLI help works."""
    parser = create_parser()
    help_text = parser.format_help()
    
    assert "honesty" in help_text
    assert "evaluate" in help_text
    assert "train" in help_text
    assert "benchmark" in help_text
    assert "dataset" in help_text
    assert "server" in help_text


def test_cli_main_help():
    """Test that main function works with help argument."""
    # Test that help doesn't crash
    try:
        result = main(["--help"])
        # Help should exit with code 0, but argparse raises SystemExit
    except SystemExit as e:
        assert e.code == 0


def test_cli_subcommands():
    """Test that all expected subcommands are available."""
    parser = create_parser()
    
    # Parse help to check subcommands are available
    help_text = parser.format_help()
    
    expected_commands = [
        "evaluate", "train", "benchmark", "dataset", "server"
    ]
    
    for command in expected_commands:
        assert command in help_text


def test_cli_verbose_flag():
    """Test that verbose flag is available."""
    parser = create_parser()
    help_text = parser.format_help()
    
    assert "--verbose" in help_text or "-v" in help_text 