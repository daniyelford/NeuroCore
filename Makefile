fmt:
	go fmt ./...

test:
	go test ./...

bench:
	go test -bench=. ./...

lint:
	golangci-lint run

build:
	go build ./...