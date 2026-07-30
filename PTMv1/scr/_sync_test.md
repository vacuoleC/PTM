# Sync Test (do not remove without confirmation)

This file exists **only** to verify the GitHub sync channel from `/data/PTM`
to `git@github.com:vacuoleC/PTM.git` works through SSH-over-HTTPS on port 443
(SSH 22 and HTTPS 443 are firewalled in this environment).

- Channel: ed25519 SSH key at `~/.ssh/id_ed25519`, routed via `~/.ssh/config`
  (`Host github.com` -> `ssh.github.com:443`)
- Fingerprint: `SHA256:r8/n/edzmBO/Bjl6epgOpPR+xAhBCiB/4wHHrq5wrBA`
- Purpose: confirm push from the remote `/data/PTM` reaches the GitHub mirror

If you can read this on `origin/main`, the channel is up.
Remove this file in a follow-up atomic commit when no longer needed.
