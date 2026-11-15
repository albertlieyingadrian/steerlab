#!/bin/sh
# The composer copies the driver's files without their mode bits: make the binary executable where the grants name it.
set -eu
for f in /opt/agentenv/drivers/ourdriver/bin/ourdriver "$(dirname "$0")/bin/ourdriver"; do
    [ -f "$f" ] && chmod 755 "$f" && xattr -c "$f" 2>/dev/null || true
done
ls -l /opt/agentenv/drivers/ourdriver/bin/ourdriver
