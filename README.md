# GenKit Handler - Agentic RAG Plugin

[![Go Reference](https://pkg.go.dev/badge/github.com/ZanzyTHEbar/genkithandler.svg)](https://pkg.go.dev/github.com/ZanzyTHEbar/genkithandler)  
[![Build Status](https://github.com/ZanzyTHEbar/genkithandler/actions/workflows/go.yml/badge.svg)](https://github.com/ZanzyTHEbar/genkithandler/actions)  
[![Coverage Status](https://coveralls.io/repos/github/ZanzyTHEbar/genkithandler/badge.svg)](https://coveralls.io/github/ZanzyTHEbar/genkithandler)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Firebase GenKit plugin that implements an Agentic Retrieval-Augmented Generation (RAG) system following the OpenAI Agentic RAG Flow specification.

## Features

✅ **Core Agentic RAG Flow** - Implements the 8-step agentic RAG process:

1. Load entire documents into context window with intelligent management
2. Chunk documents into manageable pieces respecting sentence boundaries
3. Prompt model to identify relevant chunks for queries
4. Recursively drill down into selected chunks for granular information
5. Generate responses based on retrieved information
6. Build knowledge graphs from context (optional)
7. Verify answers for factual accuracy (optional)
8. Return structured response with metadata

✅ **GenKit Integration** - Native Firebase GenKit plugin with:

- Flow definitions for agentic RAG processing
- Tool definitions for document chunking, relevance scoring, and knowledge graph extraction
- Proper request/response types with JSON schema validation

✅ **Configurable Processing** - Flexible configuration options:

- Adjustable chunk sizes and overlap
- Configurable recursive depth
- Sentence boundary respect
- Knowledge graph enablement
- Fact verification options

✅ **Knowledge Graph Construction** - Automatic entity and relation extraction:

- Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT
- Relation types: RELATED_TO, PART_OF, CAUSES, LOCATED_IN
- Confidence scoring for entities and relations

✅ **Observability** - Built-in metrics and monitoring:

- Processing time tracking
- Token usage monitoring
- Chunk processing statistics
- Recursive level trackingdler

[![Go Reference](https://pkg.go.dev/badge/github.com/ZanzyTHEbar/genkithandler.svg)](https://pkg.go.dev/github.com/ZanzyTHEbar/genkithandler)  
[![Build Status](https://github.com/ZanzyTHEbar/genkithandler/actions/workflows/go.yml/badge.svg)](https://github.com/ZanzyTHEbar/genkithandler/actions)  
[![Coverage Status](https://coveralls.io/repos/github.com/ZanzyTHEbar/genkithandler/badge.svg)](https://coveralls.io/github/ZanzyTHEbar/genkithandler)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GenKitHandler is a Go library and service framework designed to simplify interaction with multiple generative AI providers. It offers:

- Flexible provider management (e.g., Google AI, TursoDB, custom integrations)
- Primary/fallback ordering for high availability
- Structured output support
- Configuration via [Viper](https://github.com/spf13/viper)
- Custom error handling with [errbuilder-go](https://github.com/ZanzyTHEbar/errbuilder-go)
- UUID generation using [google/uuid](https://github.com/google/uuid)
- Robust, table-driven tests with [testify](https://github.com/stretchr/testify)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Provider Management**: Register, initialize, and orchestrate multiple AI providers with fallback support.
- **Structured Output**: Generate and parse JSON-like structured responses.
- **Configuration Driven**: Centralized config using Viper, supporting YAML files and environment variables.
- **Pluggable Architecture**: Easily extend or replace providers via the `AIProvider` interface.
- **Custom Error Handling**: Build rich errors with stack traces using `errbuilder-go`.
- **UUID & Logging**: Standardized identifiers and logging via `google/uuid` and domain-level logger.
- **Comprehensive Testing**: Table-driven unit tests covering core logic and edge cases.

## Installation

Ensure you have Go 1.24+ installed.

```bash
# Fetch the module
go get github.com/ZanzyTHEbar/genkithandler
```

Add to your project's `go.mod`:

```go
require github.com/ZanzyTHEbar/genkithandler v0.0.0-<latest>
```

## Configuration

GenKitHandler uses [Viper](https://github.com/spf13/viper) for configuration. It will look for `config.yaml` in the following order:

1. Current working directory
2. `./config/` subfolder
3. `$HOME/.genkithandler`
4. `/etc/genkithandler`

Example `config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8080

google_ai:
  api_key: "YOUR_GOOGLE_AI_KEY"
  default_model: "gemini-2.0-flash"
  embedding_model: "text-embedding-004"
# Additional provider and vector store settings...
```

Environment variables override file values. Prefix with `GENKITHANDLER_`, e.g., `GENKITHANDLER_GOOGLE_AI_API_KEY`.

## Quick Start

```go
package main

import (
  "context"
  "log"

  "github.com/ZanzyTHEbar/genkithandler/internal/providers"
  "github.com/ZanzyTHEbar/genkithandler/pkg/config"
  "github.com/ZanzyTHEbar/genkithandler/pkg/domain"
)

func main() {
  // Load configuration
  cfgMgr := config.NewManager()
  if err := cfgMgr.Load(); err != nil {
    log.Fatalf("config load error: %v", err)
  }
  cfg := cfgMgr.Get()

  // Initialize provider manager
  logger := domain.NewLogger(cfg.Logging)
  errHandler := domain.NewErrorHandler()
  mgr := providers.NewManager(logger, errHandler)
  if err := mgr.Initialize(context.Background(), *cfg); err != nil {
    log.Fatalf("provider init error: %v", err)
  }

  // Generate text
  result, err := mgr.GenerateText(context.Background(), "Hello, AI!")
  if err != nil {
    log.Fatalf("generation error: %v", err)
  }
  log.Println("AI Response:", result)
}
```

## Usage

1. **Register Custom Providers**: Implement the `AIProvider` interface and register via `RegisterProvider`.
2. **Primary & Fallback**: Use `SetPrimaryProvider` and inspect `GetFallbackOrder` for fine-tuned control.
3. **Structured Output**: Call `GenerateWithStructuredOutput` to unmarshal responses into typed structs.
4. **Availability Checks**: Use `IsProviderAvailable` before generating.

## Testing

```bash
# Run all tests with coverage
go test ./... -coverprofile=coverage.out
# Format and vet
go fmt ./... && go vet ./...
```

Tests are table-driven and leverage `testify` for assertions.

## Project Structure

```text
genkithandler.go         # entrypoint / package stub
cmd/                     # (future) CLI or server entry points
internal/providers/      # AI provider implementations
pkg/config/              # Configuration management
pkg/domain/              # Core domain types and utilities
Makefile                 # Common build/test commands
README.md                # Project overview
LICENSE                  # MIT license
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Run tests and ensure coverage
5. Submit a pull request

Please follow our [Coding Guidelines](CONTRIBUTING.md) and ensure all new code is accompanied by table-driven tests.

## License

This project is licensed under the [MIT License](LICENSE).
