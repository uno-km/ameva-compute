#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMEVA Compute: Standardized Remote Execution Helper
Enforces ZERO_FAKE_SUCCESS_VIOLATION=false by:
  1. Explicitly capturing remote command exit code via channel.recv_exit_status().
  2. Propagating remote rc != 0 to sys.exit(rc).
  3. Separately capturing and saving STDOUT and STDERR.
  4. Structuring raw log headers with mandatory provenance fields:
     DEVICE, COMMAND, SOURCE_COMMIT, BINARY_SHA256, MODEL_SHA256,
     START_TIMESTAMP, END_TIMESTAMP, PROCESS_EXIT_CODE, STDOUT, STDERR.
"""

import os
import sys
import time
import datetime
import hashlib
import subprocess
import paramiko

def get_local_git_commit(repo_dir):
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[remote_exec] local git commit unavailable ({e}); recorded as UNKNOWN", file=sys.stderr)
        return "UNKNOWN"

def execute_remote(
    host: str,
    user: str,
    port: int,
    command: str,
    device_name: str,
    output_raw_log_path: str = None,
    binary_remote_path: str = None,
    model_remote_path: str = None,
    binary_source_commit: str = None,
    source_binary_correspondence: str = None,
    local_manifest_sha: str = None,
    remote_manifest_sha: str = None,
    source_manifest_match: bool = None,
    timeout_sec: int = 120
):
    key_file = os.path.expanduser("~/.ssh/id_rsa")
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    source_commit = get_local_git_commit(repo_dir)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=user, key_filename=key_file, timeout=10)

    # Resolve remote binary sha256 if requested
    binary_sha256 = "N/A"
    if binary_remote_path:
        _, so, _ = ssh.exec_command(f"sha256sum {binary_remote_path} 2>/dev/null | awk '{{print $1}}'")
        b_hash = so.read().decode().strip()
        if b_hash:
            binary_sha256 = b_hash

    # Resolve remote model sha256 if requested
    model_sha256 = "N/A"
    if model_remote_path:
        _, so, _ = ssh.exec_command(f"sha256sum {model_remote_path} 2>/dev/null | awk '{{print $1}}'")
        m_hash = so.read().decode().strip()
        if m_hash:
            model_sha256 = m_hash

    start_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout_sec)
    
    stdout_bytes = stdout.read()
    stderr_bytes = stderr.read()
    remote_exit_code = stdout.channel.recv_exit_status()
    end_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    ssh.close()

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    b_commit = binary_source_commit if binary_source_commit else "UNKNOWN"
    corr = source_binary_correspondence if source_binary_correspondence else "NOT_ESTABLISHED"

    manifest_lines = ""
    if local_manifest_sha and remote_manifest_sha:
        m_match = "true" if source_manifest_match else "false"
        manifest_lines = f"LOCAL_SOURCE_MANIFEST_SHA256: {local_manifest_sha}\nREMOTE_SOURCE_MANIFEST_SHA256: {remote_manifest_sha}\nSOURCE_MANIFEST_MATCH: {m_match}\n"

    log_entry = f"""=======================================================================
DEVICE: {device_name}
COMMAND: {command.strip()}
WORKTREE_HEAD_COMMIT: {source_commit}
{manifest_lines}BINARY_SOURCE_COMMIT: {b_commit}
SOURCE_BINARY_CORRESPONDENCE: {corr}
BINARY_SHA256: {binary_sha256}
MODEL_SHA256: {model_sha256}
START_TIMESTAMP: {start_ts}
END_TIMESTAMP: {end_ts}
PROCESS_EXIT_CODE: {remote_exit_code}
=======================================================================
[STDOUT]
{stdout_text}
[STDERR]
{stderr_text}
=======================================================================
"""

    if output_raw_log_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_raw_log_path)), exist_ok=True)
        with open(output_raw_log_path, "w", encoding="utf-8") as f:
            f.write(log_entry)

    # Strictly propagate remote failure
    if remote_exit_code != 0:
        sys.stderr.write(f"REMOTE_EXECUTION_FAILED: Exit Code {remote_exit_code}\n")
        sys.stderr.write(stderr_text)
        sys.exit(remote_exit_code)

    return {
        "device": device_name,
        "exit_code": remote_exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "log_entry": log_entry
    }

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python remote_exec.py <host> <user> <port> <command> [device_name] [log_out_path]")
        sys.exit(1)
    h = sys.argv[1]
    u = sys.argv[2]
    p = int(sys.argv[3])
    cmd = sys.argv[4]
    dev = sys.argv[5] if len(sys.argv) > 5 else "GenericDevice"
    out = sys.argv[6] if len(sys.argv) > 6 else None
    res = execute_remote(h, u, p, cmd, dev, out)
    if not out:
        sys.stdout.write(res["stdout"])
        if res["stderr"]:
            sys.stderr.write(res["stderr"])

