"""Install the pinned official server or complete one-time personal OAuth login.

Usage: python -m openevolve.antigravity_acp {install,login}
"""

import argparse
import asyncio
import hashlib
import os
import platform
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from openevolve.config import LLMModelConfig
from openevolve.llm.antigravity_acp import (
    DEFAULT_BINARY,
    SUPPORTED_VERSION,
    resolve_paths,
    start_server,
    validate_profile,
)

ARCHIVE_URL = (
    "https://dl.google.com/agy-extensions/releases/linux/"
    f"agy-acp-server-{SUPPORTED_VERSION}-linux-x86_64.zip"
)
ARCHIVE_SHA256 = "9fb60956af0a9d76220a4db91ca9ac88e2a2372ad68f985ab5fceace6b825b96"


def install_server(local_archive=None):
    """Verify Google's archive and extract only the two required executables."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError(
            "Automatic ACP installation supports Linux x86_64; install manually elsewhere"
        )
    target = DEFAULT_BINARY.parent
    if target.exists():
        raise FileExistsError(f"ACP installation already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".acp-install-", dir=target.parent) as scratch:
        archive = Path(local_archive) if local_archive else Path(scratch) / "server.zip"
        digest = hashlib.sha256()
        if local_archive:
            with archive.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
        else:
            print(
                f"Downloading official Antigravity ACP {SUPPORTED_VERSION} (about 334 MB)...",
                flush=True,
            )
            with (
                urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response,
                archive.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Official ACP archive checksum mismatch; installation aborted")
        staged = Path(scratch) / SUPPORTED_VERSION
        staged.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            for name in ("agy_acp_server.par", "localharness_external"):
                with bundle.open(name) as source, (staged / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
                (staged / name).chmod(0o755)
        (staged / "archive.sha256").write_text(ARCHIVE_SHA256 + "\n")
        staged.rename(target)
    print(f"Installed: {DEFAULT_BINARY}")


async def login(executable, home, timeout=600):
    """Authenticate only; never create a generation session or send a model prompt."""
    validate_profile(home, require_auth=False)
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(home, 0o700)
    with tempfile.TemporaryDirectory(prefix="openevolve-acp-login-") as workdir:
        connection = await start_server(executable, home, workdir, allow_login=True)
        try:
            print("Complete Google sign-in in the browser if prompted.", flush=True)
            await asyncio.wait_for(connection.initialize(), timeout)
            if not (home / "antigravity-acp/acp_token.json").is_file():
                raise RuntimeError("ACP did not save persistent personal OAuth credentials")
        finally:
            await asyncio.shield(connection.close())
    print(f"ACP login complete. Dedicated profile: {home}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "login"))
    parser.add_argument("--executable", help="Official ACP executable (login only)")
    parser.add_argument("--profile", help="Dedicated GEMINI_HOME (login only)")
    parser.add_argument("--archive", help="Previously downloaded official ZIP (install only)")
    args = parser.parse_args()
    try:
        if args.command == "install":
            if args.executable or args.profile:
                parser.error("--executable and --profile apply only to login")
            install_server(args.archive)
        else:
            if args.archive:
                parser.error("--archive applies only to install")
            executable, home = resolve_paths(
                LLMModelConfig(
                    antigravity_acp_path=args.executable,
                    antigravity_acp_home=args.profile,
                )
            )
            asyncio.run(login(executable, home))
    except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
