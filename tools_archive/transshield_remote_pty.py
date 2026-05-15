import argparse
import fcntl
import os
import pty
import select
import subprocess
import sys
import termios
import time


def make_controlling_tty(slave_fd):
    def _preexec():
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)

    return _preexec


def command_run(args):
    password = os.environ.get(args.password_env, "")
    if not password:
        raise SystemExit(f"{args.password_env} is not set")
    if not args.command:
        raise SystemExit("missing command")

    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(
        args.command,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        text=False,
        preexec_fn=make_controlling_tty(slave_fd),
    )
    os.close(slave_fd)

    buffer = b""
    sent_password = False
    deadline = None if args.timeout_sec <= 0 else time.time() + args.timeout_sec
    try:
        while True:
            if deadline is not None and time.time() > deadline:
                process.kill()
                raise SystemExit(124)
            if process.poll() is not None:
                while True:
                    ready, _, _ = select.select([master_fd], [], [], 0)
                    if not ready:
                        break
                    try:
                        chunk = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                raise SystemExit(process.returncode)

            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                continue
            if not chunk:
                continue
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            buffer = (buffer + chunk)[-4096:]
            lowered = buffer.lower()
            if (b"password:" in lowered or b"passphrase" in lowered) and not sent_password:
                os.write(master_fd, (password + "\n").encode("utf-8"))
                sent_password = True
                buffer = b""
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass


def build_parser():
    parser = argparse.ArgumentParser(description="Run an interactive SSH/rsync command through a PTY.")
    parser.add_argument("--password-env", default="TRANSSHIELD_REMOTE_PASSWORD")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    parser.set_defaults(func=command_run)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    args.func(args)


if __name__ == "__main__":
    main()
