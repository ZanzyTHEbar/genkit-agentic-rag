# Go Library Makefile

# Go module path
MODULE_PATH := github.com/ZanzyTHEbar/agentic-rag

# Test settings
TEST_TIMEOUT := 30s
COVERAGE_OUT := coverage.out

# Build flags for library development
BUILD_FLAGS := -v

# Default target
.DEFAULT_GOAL := test

# Library targets - no binary building
all: fmt vet test

# Build the library (compilation check)
build:
	@echo "Building library..."
	@go build $(BUILD_FLAGS) ./...

# Format Go code
fmt:
	@echo "Formatting Go code..."
	@go fmt ./...

# Vet Go code
vet:
	@echo "Vetting Go code..."
	@go vet ./...

# Run tests with coverage
test:
	@echo "Running tests..."
	@go test -timeout $(TEST_TIMEOUT) -v ./...

# Run tests with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	@go test -timeout $(TEST_TIMEOUT) -coverprofile=$(COVERAGE_OUT) ./...
	@go tool cover -html=$(COVERAGE_OUT)

# Tidy dependencies
tidy:
	@echo "Tidying dependencies..."
	@go mod tidy

# Clean test cache and coverage files
clean:
	@echo "Cleaning..."
	@go clean -testcache
	@rm -f $(COVERAGE_OUT)

# Development workflow
dev: tidy fmt vet test

# CI workflow
ci: build test

.PHONY: all build fmt vet test test-coverage tidy clean dev ci
