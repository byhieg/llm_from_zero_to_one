#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-connect.bjb2.seetacloud.com}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-37295}"
REMOTE_DIR="${REMOTE_DIR:-/root/llm_from_zero_to_one}"
SSH_KEY="${SSH_KEY:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  tools/sync_project.sh [options]

Options:
  --host HOST          Remote host, default: connect.bjb2.seetacloud.com
  --user USER          Remote user, default: root
  --port PORT          SSH port, default: 37295
  --remote-dir DIR     Remote project directory, default: /root/llm_from_zero_to_one
  --identity PATH      SSH private key path
  --dry-run            Print rsync changes without copying
  -h, --help           Show this help

Environment variables with the same names are also supported:
  REMOTE_HOST, REMOTE_USER, REMOTE_PORT, REMOTE_DIR, SSH_KEY, DRY_RUN

Only Git-visible files are synced:
  - tracked files
  - untracked files not ignored by .gitignore

Ignored files such as .venv, __pycache__, checkpoints, train_data and swanlog are
not copied.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --user)
      REMOTE_USER="$2"
      shift 2
      ;;
    --port)
      REMOTE_PORT="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --identity)
      SSH_KEY="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required. Install rsync or use scp manually." >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SSH_ARGS=(-p "$REMOTE_PORT")
if [[ -n "$SSH_KEY" ]]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi

RSYNC_ARGS=(-az --human-readable --info=stats2,progress2 --files-from=-)
if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

echo "Syncing Git-visible files to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

ssh "${SSH_ARGS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '$REMOTE_DIR'"

git ls-files --cached --others --exclude-standard -z \
  | rsync "${RSYNC_ARGS[@]}" -e "ssh ${SSH_ARGS[*]}" --from0 ./ \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
