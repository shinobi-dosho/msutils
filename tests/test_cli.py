"""Tests for the click CLI.

The help/version paths need no MS; the MS-backed commands use the `ms` fixture
and skip when simms is unavailable.
"""
from click.testing import CliRunner

from msutils.cli import cli

COMMANDS = ["summary", "addcol", "copycol", "sumcols", "addnoise", "flagstats"]


def test_cli_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in COMMANDS:
        assert cmd in result.output


def test_cli_version():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "msutils" in result.output


def test_cli_subcommand_help():
    # Doesn't import optional deps, so always runs.
    result = CliRunner().invoke(cli, ["flagstats", "--help"])
    assert result.exit_code == 0


def test_cli_summary(ms):
    result = CliRunner().invoke(cli, ["summary", ms, "--quiet"])
    assert result.exit_code == 0, result.output


def test_cli_addcol(ms):
    result = CliRunner().invoke(cli, ["addcol", ms, "MODEL_DATA", "--clone", "DATA"])
    assert result.exit_code == 0, result.output
    assert "added" in result.output
