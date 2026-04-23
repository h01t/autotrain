- [x] Read project README and AGENTS instructions to understand the SSH dependency for remote training.
- [x] Inspect local SSH client configuration for `blackbox` and identify why public key auth is not succeeding.
- [x] Apply the minimal fix needed for key-based login on the Mac and `blackbox`.
- [ ] Verify that `ssh blackbox` works without prompting for a remote password and record the result below.

## Review

- `blackbox` already accepts the Mac's ED25519 public key (`SHA256:QkF/XO/TFdMhr2D2vytm6W66PGzmbbk1QzhffNRyUS0`).
- The failure mode is local: the private key is passphrase-protected and was not loaded into the ssh-agent / macOS Keychain.
- Updated `/Users/grmim/.ssh/config` for `blackbox` with `AddKeysToAgent yes`, `UseKeychain yes`, and `IdentitiesOnly yes`.
- `ssh-add --apple-load-keychain` reported `No identity found in the keychain`, so one interactive passphrase entry is still required to store the key.
