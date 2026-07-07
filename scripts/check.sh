#!/usr/bin/env bash

set -e

echo "==> Formatting..."
go fmt ./...

echo "==> Vet..."
go vet ./...

echo "==> Testing..."
go test ./...

echo "==> Building..."
go build ./...

echo "✓ All checks passed."