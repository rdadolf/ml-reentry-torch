#!/usr/bin/env bash
set -euo pipefail

# setup-claude.sh — Wire Claude Code config and project state inside a devcontainer.
#
# Called by postStartCommand. $PWD is the container's workspaceFolder.
# Requires HOST_WORKSPACE_FOLDER env var (set via containerEnv in devcontainer.json).
#
# See claude-code-devcontainer-fix.md in the devops repo for the full design.

if [[ -z "${HOST_WORKSPACE_FOLDER:-}" ]]; then
  echo "setup-claude: HOST_WORKSPACE_FOLDER not set — skipping" >&2
  exit 0
fi

if [[ ! -d /mnt/claude/config ]]; then
  echo "setup-claude: /mnt/claude/config not mounted — skipping" >&2
  exit 0
fi

# 1. Create ~/.claude as a local directory and symlink shared config into it.
#    Skip projects/ — it gets special handling below.
mkdir -p "$HOME/.claude/projects"
for f in /mnt/claude/config/* /mnt/claude/config/.*; do
  [[ -e "$f" ]] || continue
  name=$(basename "$f")
  # Symlink blacklist
  # settings.json caches the host's PATH; skip it so Claude Code
  # generates a fresh one with the container's actual environment.
  case "$name" in .|..|projects|settings.json) continue;; esac
  ln -sfn "$f" "$HOME/.claude/$name"
done

# 2. Symlink container project key → host project directory.
#    Claude Code derives its project key from $PWD, so the symlink target
#    must match what Claude will look up.
host_key=$(echo "$HOST_WORKSPACE_FOLDER" | tr '/' '-')
container_key=$(echo "$PWD" | tr '/' '-')
ln -sfn "/mnt/claude/projects/$host_key" "$HOME/.claude/projects/$container_key"

# 3. Write path-map.md for cross-environment path translation.
#    Persists on the host (via the symlink) so host-side sessions can
#    interpret container paths in shared session history.
mkdir -p "$HOME/.claude/projects/$container_key/memory"
cat > "$HOME/.claude/projects/$container_key/memory/path-map.md" << EOF
Path equivalence: ${HOST_WORKSPACE_FOLDER} (host) = ${PWD} (container).
These refer to the same workspace. Translate paths accordingly when reading
session history or file references from the other environment.
EOF

echo "setup-claude: linked container project key ($container_key) → host ($host_key)"
