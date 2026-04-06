import os
import subprocess
from pathlib import Path
import argparse

# ====== KONFIGURASI DEFAULT ======
DEFAULT_REMOTE_USER = "yanoar"
DEFAULT_REMOTE_HOST = "10.10.240.17"
DEFAULT_REMOTE_BASE = "/clusterfs/staff/yanoar/ml_qemV3"
DEFAULT_LOCAL_BASE = "/home/fadhil/ml_qemV2/"
DEFAULT_OUTPUT_DIR = "unified_output_lupa"
DEFAULT_SSH_KEY_PATH = str(Path.home() / ".ssh" / "id_rsa")
# =================================


def generate_ssh_key(ssh_key_path: str):
    if Path(ssh_key_path).exists():
        print(f"SSH key sudah ada di {ssh_key_path}, skip generate.")
        return

    cmd = ["ssh-keygen", "-t", "rsa", "-b", "4096", "-f", ssh_key_path, "-N", ""]
    subprocess.run(cmd, check=True)
    print(f"SSH key dibuat di {ssh_key_path}")


def rsync_call(src: str, dest: str, ssh_key_path: str):
    cmd = ["rsync", "-avz", "--exclude", ".venv/","--exclude", "unified_output_*/", "--exclude", "hasil_eksperimen*/", "-e", f"ssh -i {ssh_key_path}", src, dest]
    subprocess.run(cmd, check=True)


def push_local_to_remote(local_base, remote_user, remote_host, remote_base, ssh_key_path):
    src = f"{local_base}/"
    dest = f"{remote_user}@{remote_host}:{remote_base}/"
    rsync_call(src, dest, ssh_key_path)
    print("Sync local -> remote selesai.")


def pull_remote_to_local(local_base, remote_user, remote_host, remote_base, ssh_key_path):
    src = f"{remote_user}@{remote_host}:{remote_base}/"
    dest = f"{local_base}/"
    rsync_call(src, dest, ssh_key_path)
    print("Sync remote -> local selesai.")


def pull_output_only(local_base, remote_user, remote_host, remote_base, ssh_key_path, output_dir):
    src = f"{remote_user}@{remote_host}:{remote_base}/{output_dir}/"
    dest = f"{local_base}/{output_dir}/"
    os.makedirs(dest, exist_ok=True)
    rsync_call(src, dest, ssh_key_path)
    print("Sync output remote -> local selesai.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Tool SSH key + rsync (local<->remote + folder output)."
    )

    parser.add_argument("--remote-user", default=DEFAULT_REMOTE_USER)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-base", default=DEFAULT_LOCAL_BASE)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY_PATH)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("gen-key", help="Generate SSH key RSA 4096 bit.")
    sub.add_parser("push", help="Sync local -> remote (folder kerja penuh).")
    sub.add_parser("pull", help="Sync remote -> local (folder kerja penuh).")
    sub.add_parser("pull-output", help="Sync hanya folder output dari remote -> local.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "gen-key":
        generate_ssh_key(args.ssh_key)
    elif args.command == "push":
        push_local_to_remote(args.local_base, args.remote_user, args.remote_host,
                             args.remote_base, args.ssh_key)
    elif args.command == "pull":
        pull_remote_to_local(args.local_base, args.remote_user, args.remote_host,
                             args.remote_base, args.ssh_key)
    elif args.command == "pull-output":
        pull_output_only(args.local_base, args.remote_user, args.remote_host,
                         args.remote_base, args.ssh_key, args.output_dir)


if __name__ == "__main__":
    main()
