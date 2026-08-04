"""Tests for the click CLI.

The help and version paths need no MS; the rest run against the synthetic MS
from the `ms` fixture.
"""
import json

import pytest
from click.testing import CliRunner

from msutils.cli import cli

#: Every command the CLI exposes. Kept exhaustive so a new command that is
#: never wired into the group is caught here.
COMMANDS = [
    "info", "summary", "addcol", "delcol", "renamecol", "copycol", "sumcols",
    "addnoise", "subset", "average", "flagstats", "flags", "du", "check",
    "taql", "convert",
]


def _run(*args):
    return CliRunner().invoke(cli, list(args))


def test_cli_help_lists_every_command():
    result = _run("--help")
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output, command


def test_no_undocumented_commands():
    """The listing and COMMANDS must not drift apart."""
    from msutils.cli import cli as group
    assert sorted(group.commands) == sorted(COMMANDS)


def test_cli_version():
    result = _run("--version")
    assert result.exit_code == 0
    assert "msutils" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_has_help(command):
    # Help must not require the optional extras.
    result = _run(command, "--help")
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------
# metadata

def test_cli_info(ms):
    result = _run("info", ms)
    assert result.exit_code == 0, result.output
    assert "MEERKAT" in result.output
    assert "PKS1934-638" in result.output


def test_cli_info_verbose(ms):
    result = _run("info", ms, "-v")
    assert result.exit_code == 0, result.output
    assert "Antennas (4)" in result.output


def test_cli_info_json_stdout(ms):
    result = _run("info", ms, "--json-stdout")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["format"] == "MSv2"


def test_cli_info_json_file(ms, tmp_path):
    out = tmp_path / "info.json"
    result = _run("info", ms, "--json", str(out))
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["nrows"] == 216


def test_cli_info_levels(ms):
    for level in ("meta", "full", "data"):
        result = _run("info", ms, "--level", level)
        assert result.exit_code == 0, result.output


def test_cli_summary_is_deprecated(ms):
    result = _run("summary", ms, "--quiet")
    assert result.exit_code == 0, result.output
    assert "deprecated" in result.output


# --------------------------------------------------------------------------
# columns

def test_cli_addcol(ms):
    result = _run("addcol", ms, "MODEL_DATA", "--clone", "DATA")
    assert result.exit_code == 0, result.output
    assert "added" in result.output


def test_cli_delcol(ms):
    _run("addcol", ms, "MODEL_DATA", "--clone", "DATA")
    result = _run("delcol", ms, "MODEL_DATA")
    assert result.exit_code == 0, result.output
    assert "MODEL_DATA" in result.output


def test_cli_delcol_protects_required_columns(ms):
    result = _run("delcol", ms, "TIME")
    assert result.exit_code != 0


def test_cli_renamecol(ms):
    _run("copycol", ms, "DATA", "MODEL_DATA")
    result = _run("renamecol", ms, "MODEL_DATA", "OLD_MODEL")
    assert result.exit_code == 0, result.output
    assert "OLD_MODEL" in result.output


def test_cli_copycol(ms):
    result = _run("copycol", ms, "DATA", "CORRECTED_DATA")
    assert result.exit_code == 0, result.output


def test_cli_sumcols(ms):
    _run("copycol", ms, "DATA", "CORRECTED_DATA")
    result = _run("sumcols", ms, "DATA", "CORRECTED_DATA", "--out", "MODEL_DATA")
    assert result.exit_code == 0, result.output


def test_cli_sumcols_subtract_needs_two_columns(ms):
    result = _run("sumcols", ms, "DATA", "--out", "X", "--subtract")
    assert result.exit_code != 0
    assert "exactly two" in result.output


def test_cli_addnoise(ms):
    _run("addcol", ms, "MODEL_DATA", "--clone", "DATA")
    result = _run("addnoise", ms, "--column", "MODEL_DATA", "--noise", "1.0")
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------
# datasets

def test_cli_subset(ms, tmp_path):
    out = str(tmp_path / "sub.ms")
    result = _run("subset", ms, out, "--field", "DEEP_2")
    assert result.exit_code == 0, result.output
    assert _run("check", out).exit_code == 0


def test_cli_average(ms, tmp_path):
    pytest.importorskip("africanus.averaging",
                        reason="codex-africanus not installed")
    out = str(tmp_path / "avg.ms")
    result = _run("average", ms, out, "--chan-bin", "2")
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------
# flags

def test_cli_flagstats(ms):
    result = _run("flagstats", ms)
    assert result.exit_code == 0, result.output
    assert "Flag statistics" in result.output


def test_cli_flagstats_json(ms, tmp_path):
    out = tmp_path / "flags.json"
    result = _run("flagstats", ms, "--json", str(out), "--quiet")
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["total"]["total"] > 0


def test_cli_flag_versions(ms):
    assert _run("flags", "backup", ms, "--name", "v1").exit_code == 0
    listing = _run("flags", "list", ms)
    assert "v1" in listing.output
    assert _run("flags", "restore", ms, "v1").exit_code == 0
    assert _run("flags", "delete", ms, "v1").exit_code == 0
    assert "no saved flag versions" in _run("flags", "list", ms).output


# --------------------------------------------------------------------------
# diagnostics

def test_cli_du(ms):
    result = _run("du", ms)
    assert result.exit_code == 0, result.output
    assert "Disk usage" in result.output


def test_cli_check_passes(ms):
    result = _run("check", ms)
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_cli_check_exits_nonzero_on_error(ms):
    _run("delcol", ms, "ARRAY_ID", "--force")
    result = _run("check", ms)
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_cli_taql(ms):
    result = _run("taql", "SELECT DISTINCT FIELD_ID FROM $1", "--ms", ms)
    assert result.exit_code == 0, result.output
    assert "FIELD_ID" in result.output
