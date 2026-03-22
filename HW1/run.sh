#!/bin/bash
exec env -i \
  HOME="$HOME" \
  PATH="/usr/local/bin:/usr/bin:/bin" \
  DISPLAY="$DISPLAY" \
  XAUTHORITY="$XAUTHORITY" \
  XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
  WAYLAND_DISPLAY="$WAYLAND_DISPLAY" \
  python3 /home/howardchen/HW1/main.py "$@"
