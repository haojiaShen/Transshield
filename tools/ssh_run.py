#!/usr/bin/env python3
"""SSH helper: run a command on the remote server via paramiko."""
import sys, paramiko, getpass

HOST = "10.204.248.175"
PORT = 9001
USER = "wyb"
PASS = "wyb123"

def run_remote(cmd, timeout=600):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    client.close()
    return rc, out, err

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo hello"
    rc, out, err = run_remote(cmd)
    if out: print(out, end="")
    if err: print(err, end="", file=sys.stderr)
    sys.exit(rc)
