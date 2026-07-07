fmt:
	go fmt ./...

vet:
	go vet ./...

test:
	go test ./...

bench:
	go test -bench=. -benchmem ./...

coverage:
	go test ./... -coverprofile=coverage.out
	go tool cover -func=coverage.out

lint:
	golangci-lint run

build:
	go build ./...

check: fmt vet test build