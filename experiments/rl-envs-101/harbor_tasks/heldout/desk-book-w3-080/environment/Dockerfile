FROM python:3.11-slim@sha256:61e01f72680f86a5279ad510818ab7b2ee9f049822589d4e34e33e044360356c
# tmux and asciinema: the terminal agent (terminus-2) drives a tmux session; installed at build time, since the
# sandbox itself has no network
RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY desk.py desk_cli.py world.json /opt/desk/
RUN printf '#!/bin/sh\nexec python3 /opt/desk/desk_cli.py "$@"\n' > /usr/local/bin/desk && chmod +x /usr/local/bin/desk \
    && python3 /opt/desk/desk_cli.py --init /app/clinic.db && rm /opt/desk/world.json \
    && useradd --create-home --shell /bin/bash agent && chown -R agent:agent /app
