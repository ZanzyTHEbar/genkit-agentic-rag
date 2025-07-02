# GenKit Handler - Agentic RAG Plugin

[![Go Reference](https://pkg.go.dev/badge/github.com/ZanzyTHEbar/genkithandler.svg)](https://pkg.go.dev/github.com/ZanzyTHEbar/genkithandler)
[![Build Status](https://github.com/ZanzyTHEbar/genkithandler/actions/workflows/go.yml/badge.svg)](https://github.com/ZanzyTHEbar/genkithandler/actions)
[![Coverage Status](https://coveralls.io/repos/github.com/ZanzyTHEbar/genkithandler/badge.svg)](https://coveralls.io/github/ZanzyTHEbar/genkithandler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Firebase GenKit plugin that implements an Agentic Retrieval-Augmented Generation (RAG) system following the OpenAI Agentic RAG Flow specification.

## Features

- **Agentic RAG Flow**: Implements the 8-step agentic RAG process:
  1. Load entire documents into context window
  2. Chunk documents into manageable pieces (sentence boundaries)
  3. Prompt model to identify relevant chunks
  4. Recursively drill down into selected chunks
  5. Generate responses from retrieved information
  6. Build knowledge graphs (optional)
  7. Verify answers for factual accuracy (optional)
  8. Return structured response with metadata
- **GenKit Integration**: Native Firebase GenKit plugin with flows and tools for chunking, scoring, and knowledge graph extraction
- **Configurable**: Chunk size, recursive depth, knowledge graph, and fact verification options
- **Observability**: Processing time, chunk stats, and recursive level tracking

## Quick Start

### Installation

```bash
go get github.com/ZanzyTHEbar/genkithandler
```

### Example Usage

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/genkit"
	"github.com/ZanzyTHEbar/genkithandler"
	"github.com/ZanzyTHEbar/genkithandler/internal/agentic"
)

func main() {
	ctx := context.Background()
	g, err := genkit.Init(ctx)
	if err != nil {
		log.Fatalf("Failed to initialize GenKit: %v", err)
	}

	if err := genkithandler.InitializeAgenticRAGWithDefaults(g); err != nil {
		log.Fatalf("Failed to initialize agentic RAG plugin: %v", err)
	}

	fmt.Println("Agentic RAG plugin initialized successfully!")

	config := genkithandler.DefaultAgenticRAGConfig()
	processor := genkithandler.NewAgenticRAGProcessor(config)

	request := agentic.AgenticRAGRequest{
		Query: "What are the key components of artificial intelligence?",
		Documents: []string{
			"Artificial intelligence (AI) consists of several key components including machine learning, natural language processing, computer vision, robotics, and expert systems. Machine learning enables systems to learn from data without explicit programming. Natural language processing allows computers to understand and generate human language. Computer vision gives machines the ability to interpret visual information. Robotics combines AI with physical systems to create autonomous agents. Expert systems capture and utilize domain-specific knowledge to solve complex problems.",
			"The field of AI has evolved significantly since its inception. Early AI focused on symbolic reasoning and rule-based systems. Modern AI emphasizes data-driven approaches, particularly deep learning and neural networks. These approaches have revolutionized applications in image recognition, speech processing, and game playing. The integration of big data and powerful computing resources has accelerated AI development across industries.",
		},
		Options: agentic.AgenticRAGOptions{
			MaxChunks:              20,
			RecursiveDepth:         3,
			EnableKnowledgeGraph:   true,
			EnableFactVerification: true,
			Temperature:            0.7,
		},
	}

	response, err := processor.Process(ctx, request)
	if err != nil {
		log.Fatalf("Failed to process agentic RAG request: %v", err)
	}

	fmt.Printf("\n=== Agentic RAG Response ===\n")
	fmt.Printf("Answer: %s\n\n", response.Answer)
	fmt.Printf("Processing Metadata:\n")
	fmt.Printf("- Processing Time: %v\n", response.ProcessingMetadata.ProcessingTime)
	fmt.Printf("- Chunks Processed: %d\n", response.ProcessingMetadata.ChunksProcessed)
	fmt.Printf("- Recursive Levels: %d\n", response.ProcessingMetadata.RecursiveLevels)
	fmt.Printf("- Model Calls: %d\n", response.ProcessingMetadata.ModelCalls)
	fmt.Printf("- Tokens Used: %d\n\n", response.ProcessingMetadata.TokensUsed)

	fmt.Printf("Relevant Chunks (%d):\n", len(response.RelevantChunks))
	for i, chunk := range response.RelevantChunks {
		fmt.Printf("  %d. %s (Score: %.3f)\n", i+1, chunk.Chunk.Content[:min(100, len(chunk.Chunk.Content))]+"...", chunk.Chunk.RelevanceScore)
	}

	if response.KnowledgeGraph != nil {
		fmt.Printf("\nKnowledge Graph:\n")
		fmt.Printf("- Entities: %d\n", len(response.KnowledgeGraph.Entities))
		fmt.Printf("- Relations: %d\n", len(response.KnowledgeGraph.Relations))
		fmt.Printf("\nTop Entities:\n")
		for i, entity := range response.KnowledgeGraph.Entities {
			if i >= 5 {
				break
			}
			fmt.Printf("  - %s (%s) [%.2f]\n", entity.Name, entity.Type, entity.Confidence)
		}
	}

	if response.FactVerification != nil {
		fmt.Printf("\nFact Verification:\n")
		fmt.Printf("- Overall Status: %s\n", response.FactVerification.Overall)
		fmt.Printf("- Claims Verified: %d/%d\n",
			countVerifiedClaims(response.FactVerification.Claims),
			len(response.FactVerification.Claims))
	}

	fmt.Println("\n=== Example completed successfully! ===")
}

func countVerifiedClaims(claims []agentic.Claim) int {
	count := 0
	for _, claim := range claims {
		if claim.Status == "verified" {
			count++
		}
	}
	return count
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
```

## API Reference

### Core Types

#### `AgenticRAGRequest`

```go
type AgenticRAGRequest struct {
    Query     string            `json:"query"`
    Documents []string          `json:"documents,omitempty"`
    Options   AgenticRAGOptions `json:"options,omitempty"`
}
```

#### `AgenticRAGResponse`

```go
type AgenticRAGResponse struct {
    Answer             string             `json:"answer"`
    RelevantChunks     []ProcessedChunk   `json:"relevant_chunks"`
    KnowledgeGraph     *KnowledgeGraph    `json:"knowledge_graph,omitempty"`
    FactVerification   *FactVerification  `json:"fact_verification,omitempty"`
    ProcessingMetadata ProcessingMetadata `json:"processing_metadata"`
}
```

### GenKit Flows

- **`agenticRAG`** - Main agentic RAG processing flow
  - Input: `AgenticRAGRequest`
  - Output: `AgenticRAGResponse`

### GenKit Tools

- **`chunkDocument`** - Document chunking tool
- **`scoreRelevance`** - Relevance scoring tool
- **`extractKnowledgeGraph`** - Knowledge graph extraction tool

## Development Status

This is a **Minimal Viable Prototype (MVP)** implementation that provides:

- Core agentic RAG flow according to specification
- GenKit plugin integration
- Document chunking with sentence boundary respect
- Recursive chunk refinement
- Basic knowledge graph construction
- Simple fact verification
- Comprehensive observability metrics

### Limitations (To Be Implemented)

- Real LLM integration (currently uses placeholder logic)
- Advanced prompt engineering and templating
- Vector embedding and similarity search
- External fact verification sources
- Multi-agent orchestration
- Streaming responses
- Advanced error handling and retry logic

## Example

See [`example/main.go`](example/main.go) for a complete working example.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.
