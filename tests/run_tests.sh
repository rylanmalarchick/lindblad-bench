#!/usr/bin/env bash
# Run all tests. Build first if needed.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

if [ ! -d "$ROOT/build" ]; then
    echo "Building..."
    cmake -B "$ROOT/build" -S "$ROOT" -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$ROOT/build" -- -j"$(nproc)" > /dev/null
fi

echo "=== Running tests ==="
cd "$ROOT/build"
ctest --output-on-failure
