#!/usr/bin/env bash

cd "$(dirname $0)"/..
poetry run covid19pr \
  --config-file config/localhost.toml \
  --assets-dir assets \
  --output-dir output \
  $*