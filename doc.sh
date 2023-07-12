#!/bin/bash
function print_m {
  echo -e "\033[0;97m$1\033[m"
}

function ask_v {
  echo -n -e "\033[0;97m$1\033[m " && read "$2"
}


if [[ "$1" == "--build" ]]; then
  echo "Building..."
  python3 -m sphinx ./ docs/_build
  exit 0
fi

if [[ "$1" == "--help" ]]; then
  echo "$0 [--build]"
  exit 0
fi

if [[ -z "$JANNE_DOC_PORT" ]]; then
  JANNE_DOC_PORT=8000;
fi;

print_m "Welcome to the JANNE documentation helper"
print_m "Choose an action to run"
print_m " 1) Build"
print_m " 2) Serve"
print_m " 3) Quit"

ACTION=0

while true; do

  ask_v "Choose an action :" "ACTION"

  if [[ "$ACTION" -eq 1 ]]; then python3 -m sphinx ./ docs/_build; fi
  if [[ "$ACTION" -eq 2 ]]; then
    python3 -m http.server --directory docs/_build "$JANNE_DOC_PORT"
    print_m "Server closed"
  fi
  if [[ "$ACTION" -eq 3 ]]; then exit 0; fi

done

