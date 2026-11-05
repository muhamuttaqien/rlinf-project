# Administrator-Managed Observation Deployment Plan

This proposal prepares a serial-disabled qualification service on the NUC. It requires administrator review and explicit installation authorization. No commands in the installation or startup sections have been executed. Physical access requires a separate plan and onsite authorization.

## Privilege Boundary

Use the frozen Milestone 2A observation-only executable, not the current development checkout, which contains later offline command work. The candidate was staged as an ordinary user at `/tmp/fr3_m2b_admin_review_2eqvpi33`. Its original build is `/tmp/rlinf_robotiq_m2a_3krbaoaq`. CMake installation into the staging directory removed the original executable's unsafe `/opt/ros/humble/lib:` RUNPATH (the empty component searches the working directory).

Install reviewed native executables as root-owned, mode 0755, under `/opt/rlinf/robotiq-observation/m2a-20261027`. Run them as the dedicated nonlogin account `rlinf-robotiq` through the reviewed systemd unit. Never run the staging binaries with sudo. Use administrator-owned system utilities to copy data, verify installed hashes, and manage this single service. No sudoers rule, setuid binary, file capability, udev change or global procfs permission change is proposed.

The service receives only `CAP_DAC_READ_SEARCH` and `CAP_SYS_PTRACE`. These address directory and procfs descriptor inspection checks; actual sufficiency remains unverified under the target security policy. They permit broad host inspection, including potentially sensitive information. Seccomp denies `ptrace`, `process_vm_readv`, `process_vm_writev` and `kcmp`; this is defense in depth, not a device-scoped inspection privilege. Administrator acceptance of the remaining exposure is required. If unacceptable, stop and design a separately reviewed supervisor rather than weakening `PortClaim`.

The unit retains the host process view, shared `/tmp` lock inode, fail-closed scan, and original vendor `TIOCEXCL` implementation. It grants no `CAP_SYS_ADMIN`. Its strict device policy permits only null/zero/random/urandom, excluding the physical serial adapter. Configuration and CLI both force `enabled=false`. It has no boot-enable section, no restart, and a 30-second runtime bound. It does not support physical qualification as written.

`port-claim-check` uses the frozen `PortClaim` to stat the fixed adapter, acquire the cooperative lock, inspect descriptors and release its own claim. It never constructs a serial connection. A successful preflight is an instantaneous ownership observation, not a lasting certificate: the live bridge must retain its own gate in any future separately reviewed serial-enabled deployment.

## Reviewed Candidate Identities

Verify these hashes on the installed files before any execution. Root-owned ancestors, executable libraries and configuration must also be non-writable by unprivileged users; resolve symlinks and inspect ACLs. The dependency check resolved libraries as an ordinary user but is not a complete privileged dependency trust audit.

| File | SHA-256 |
| --- | --- |
| Installed observation executable | `3cd0c07c9621e7683038e1e489838209c1218d40a5322dcf01f85c625490ce57` |
| Installed preflight executable | `23c01f3477b06133d576dc3b7caf15145d7bae7285682aa9ec277321130b8dc8` |
| `observation-disabled.yaml` | `66d4a291ea6f2933fc3b48b31750c2f98f5fe732373e55baa1505856dbea2e86` |
| `rlinf-robotiq-observation.service` | `a6cd44732bf4dd1d4936b46c319df564ded8c41bf75d6c50a9a84aacabdaca36` |

The original pre-install node hash is `b005b04bf7e2ac8484d3e412d22df1aac56aa1875d1419abdefe633176a40ce8`. The private patched vendor archives remain linked into both native candidates. See the report's evidence for source and build hashes.

## Installation After Authorization

An administrator must use an authenticated NUC terminal; do not submit passwords through chat. Run each command only if the preceding check succeeds. Stop on any mismatch or unexpected existing identity/path. Review the proposal files and hashes independently before copying from the account-writable staging area.

First inspect existing identities, paths, ACLs and the adapter's metadata. These commands do not open the device. `getent` should find neither new identity; unexpected existing entries require review. Existing parents must be administrator-owned and protected, including resolved library paths under `/opt/ros/humble` and system library paths. Use `getfacl -p` where available and stop if effective unprivileged write permissions remain.

```bash
getent passwd rlinf-robotiq
getent group rlinf-robotiq
sudo /usr/bin/namei -l /opt/rlinf/robotiq-observation/m2a-20261027
sudo /usr/bin/namei -l /etc/rlinf/robotiq-observation/observation-disabled.yaml
sudo /usr/bin/namei -l /etc/systemd/system/rlinf-robotiq-observation.service
sudo /usr/bin/namei -l /opt/ros/humble/lib
/usr/bin/readlink -f /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0
/usr/bin/stat -Lc '%F %t:%T' /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DA95C62Y-if00-port0
sudo /usr/bin/stat /tmp/rlinf-robotiq-188-1.lock
```

The last-known device number is 188:1 (`bc:1` in hexadecimal). Stop if it changes, identity validation fails, or an existing lock needs review. Never replace or unlink an existing lock inode. An empty `fuser` result is insufficient. The following assumes the dedicated identity, release/config/unit paths and lock are absent and that reviewed parent directories are safe. Creating the shared lock requires an administrator-controlled maintenance interval so another cooperating launch cannot race its creation.

```bash
sudo /usr/sbin/groupadd --system rlinf-robotiq
sudo /usr/sbin/useradd --system --gid rlinf-robotiq --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin rlinf-robotiq
sudo /usr/bin/install -d -o root -g root -m 0755 /opt/rlinf/robotiq-observation/m2a-20261027/lib/rlinf_robotiq_observation /etc/rlinf/robotiq-observation
sudo /usr/bin/install -o root -g root -m 0755 /tmp/fr3_m2b_admin_review_2eqvpi33/release/lib/rlinf_robotiq_observation/robotiq_observation /opt/rlinf/robotiq-observation/m2a-20261027/lib/rlinf_robotiq_observation/robotiq_observation
sudo /usr/bin/install -o root -g root -m 0755 /tmp/fr3_m2b_admin_review_2eqvpi33/release/lib/rlinf_robotiq_observation/port-claim-check /opt/rlinf/robotiq-observation/m2a-20261027/lib/rlinf_robotiq_observation/port-claim-check
sudo /usr/bin/install -o root -g root -m 0644 /tmp/fr3_m2b_admin_review_2eqvpi33/proposal/observation-disabled.yaml /etc/rlinf/robotiq-observation/observation-disabled.yaml
sudo /usr/bin/install -o root -g root -m 0644 /tmp/fr3_m2b_admin_review_2eqvpi33/proposal/rlinf-robotiq-observation.service /etc/systemd/system/rlinf-robotiq-observation.service
sudo /usr/bin/install -o root -g root -m 0644 /dev/null /tmp/rlinf-robotiq-188-1.lock
sudo /usr/bin/sha256sum /opt/rlinf/robotiq-observation/m2a-20261027/lib/rlinf_robotiq_observation/robotiq_observation /opt/rlinf/robotiq-observation/m2a-20261027/lib/rlinf_robotiq_observation/port-claim-check /etc/rlinf/robotiq-observation/observation-disabled.yaml /etc/systemd/system/rlinf-robotiq-observation.service
sudo /usr/bin/namei -l /opt/rlinf/robotiq-observation/m2a-20261027/lib/rlinf_robotiq_observation/robotiq_observation
sudo /usr/bin/namei -l /etc/rlinf/robotiq-observation/observation-disabled.yaml
sudo /usr/bin/namei -l /etc/systemd/system/rlinf-robotiq-observation.service
sudo /usr/bin/systemd-analyze verify --man=no /etc/systemd/system/rlinf-robotiq-observation.service
```

Compare all four installed hashes literally with the table. Check the new account has no supplementary groups and no sudo rights. Inspect installed executable and resolved dependency ownership/ACLs, including runtime-loaded Fast DDS and message type-support libraries. Stop before launch unless this trust review and static verification pass. Do not source account-writable ROS setup scripts as root. Any stale staging files or unavailable binaries require restaging and renewed identity review, not substitution of the current development executable.

## Bounded Startup After Authorization

Only after the reviewed files are installed and administrator approval includes serial-disabled startup, load and start this unit. Do not enable it at boot. Its preflight must complete under the same service identity/capabilities and must reject inaccessible process descriptors or an existing adapter owner.

```bash
sudo /usr/bin/systemctl daemon-reload
sudo /usr/bin/systemctl start rlinf-robotiq-observation.service
sudo /usr/bin/systemctl show rlinf-robotiq-observation.service -p User -p Group -p MainPID -p ExecStartPre -p ExecStart -p CapabilityBoundingSet -p AmbientCapabilities -p DevicePolicy -p DeviceAllow -p ActiveState -p Result
sudo /usr/bin/journalctl -u rlinf-robotiq-observation.service -n 80 --no-pager
sudo /usr/bin/systemctl stop rlinf-robotiq-observation.service
sudo /usr/bin/systemctl show rlinf-robotiq-observation.service -p MainPID -p ActiveState -p Result
```

Inspect and stop promptly; 30 seconds is a backstop, and expiration may mark the unit failed. Record the preflight exit status, effective process capabilities, visible host procfs, device policy and clean shutdown. No ROS topic subscription or repeated DDS test is needed. A failed preflight is a blocker, not grounds to expand privileges or ignore processes. No serial-enabled launch is authorized by these commands.

## Rollback After Authorization

Stop only this service and remove only files created by this deployment. Preserve logs and evidence. Run the removal commands only after confirming the paths still belong solely to this installation; retain shared parents and the harmless lock inode to avoid disrupting any other holder.

```bash
sudo /usr/bin/systemctl stop rlinf-robotiq-observation.service
sudo /usr/bin/rm -- /etc/systemd/system/rlinf-robotiq-observation.service /etc/rlinf/robotiq-observation/observation-disabled.yaml
sudo /usr/bin/rm -r -- /opt/rlinf/robotiq-observation/m2a-20261027
sudo /usr/bin/systemctl daemon-reload
sudo /usr/bin/systemctl reset-failed rlinf-robotiq-observation.service
```

Remove the dedicated account/group with `sudo /usr/sbin/userdel rlinf-robotiq` and `sudo /usr/sbin/groupdel rlinf-robotiq` only if created here and no processes or other deployments use them. Account removal can leave numeric ownership on retained logs; record it. There are no proposed changes to FR3, vendor installations, global ROS libraries, sudo policy or device permissions to undo.
