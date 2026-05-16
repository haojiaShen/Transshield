#!/usr/bin/env python3
"""Upload a file to the remote server via paramiko."""
import sys, paramiko

HOST = "10.204.248.175"
PORT = 9001
USER = "wyb"
PASS = "wyb123"

def upload_file(local_path, remote_path):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    client.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <local_path> <remote_path>", file=sys.stderr)
        sys.exit(1)
    upload_file(sys.argv[1], sys.argv[2])
    print(f"Uploaded {sys.argv[1]} to {sys.argv[2]}")
