import subprocess
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
def lint():
    return subprocess.run(["ruff", "check", "."]).returncode


@cli.command()
def format_check():
    return subprocess.run(["ruff", "format", ".", "--check"]).returncode


@cli.command()
def format():
    return subprocess.run(["ruff", "format", "."]).returncode


@cli.command()
def fix():
    lint_res = subprocess.run(["ruff", "check", ".", "--fix"]).returncode
    format_res = subprocess.run(["ruff", "format", "."]).returncode
    return lint_res or format_res


@cli.command()
def install_hooks():
    hooks_dir = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    hook_path = Path(hooks_dir) / "pre-commit"
    if hook_path.exists():
        hook_path.unlink()

    hook_content = """#!/bin/sh
    python scripts/dev.py fix
    """
    hook_path.write_text(hook_content)
    hook_path.chmod(0o755)


if __name__ == "__main__":
    cli()
