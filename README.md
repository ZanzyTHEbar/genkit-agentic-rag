# Genkit Agentic RAG - Advanced Agentic RAG Plugin

<!-- [![Go Reference](https://pkg.go.dev/badge/github.com/ZanzyTHEbar/agentic-rag.svg)](https://pkg.go.dev/github.com/ZanzyTHEbar/agentic-rag)
[![Build Status](https://github.com/ZanzyTHEbar/agentic-rag/actions/workflows/go.yml/badge.svg)](https://github.com/ZanzyTHEbar/agentic-rag/actions)
[![Coverage Status](https://coveralls.io/repos/github/ZanzyTHEbar/agentic-rag/badge.svg)](https://coveralls.io/github/ZanzyTHEbar/agentic-rag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) -->

A production-ready Firebase GenKit plugin that implements an Agentic Retrieval-Augmented Generation (RAG) system following the OpenAI Agentic RAG Flow specification with sophisticated reasoning capabilities.

## Features

### 🚀 Advanced Agentic RAG Flow

- **LLM-powered chunk relevance scoring**: Real language model analysis for intelligent chunk selection
- **Sophisticated response generation**: Advanced prompt engineering with comprehensive context awareness
- **Recursive drilling**: Deep document analysis with configurable depth limits
- **Knowledge graph construction**: Automatic entity and relationship extraction with confidence scoring
- **Fact verification**: Claim-by-claim verification against source documents

### 🧠 Smart Document Processing

- **Sentence-aware chunking**: Respects natural language boundaries
- **Context-preserving**: Maintains document structure and relationships
- **Adaptive chunk sizing**: Optimizes for model token limits and processing efficiency
- **Multi-document support**: Handles complex document collections

### 🔧 Full GenKit Integration

- **Streaming support**: Ready for real-time response streaming
- **Multiple model support**: Works with any GenKit-compatible LLM
- **Configuration flexibility**: Comprehensive options for fine-tuning behavior
- **Error resilience**: Robust fallback mechanisms for production reliability

### 📊 Observability & Metrics

- **Processing time tracking**: Detailed performance metrics
- **Token usage monitoring**: Track LLM API costs and efficiency
- **Confidence scoring**: All outputs include confidence assessments
- **Recursive depth tracking**: Monitor analysis complexity

## Quick Start

### Installation

```bash
go get github.com/ZanzyTHEbar/genkit-agentic-rag
```
### Example Usage
```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/googlegenai"
    "github.com/ZanzyTHEbar/genkit-agentic-rag"
    "github.com/ZanzyTHEbar/genkit-agentic-rag/plugin"
)

func main() {
    ctx := context.Background()

    // Initialize GenKit with Google AI
    g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
    if err != nil {
        log.Fatalf("Failed to initialize GenKit: %v", err)
    }

    // Configure advanced Agentic RAG
    config := &plugin.AgenticRAGConfig{
        Genkit:    g,
        ModelName: "googleai/gemini-2.5-flash",
        Processing: plugin.ProcessingConfig{
            DefaultChunkSize:      800,
            DefaultMaxChunks:      25,
            DefaultRecursiveDepth: 4,
            RespectSentences:      true,
        },
        KnowledgeGraph: plugin.KnowledgeGraphConfig{
            Enabled:                true,
            EntityTypes:            []string{"PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT"},
            RelationTypes:          []string{"DEVELOPS", "USES", "FOUNDED", "LOCATED_IN"},
            MinConfidenceThreshold: 0.8,
        },
        FactVerification: plugin.FactVerificationConfig{
            Enabled:              true,
            RequireEvidence:      true,
            MinConfidenceScore:   0.7,
        },
    }

    // Initialize plugin
    if err := agentic-rag.InitializeAgenticRAG(g, config); err != nil {
        log.Fatalf("Failed to initialize agentic RAG: %v", err)
    }

    // Create processor
    processor := agentic-rag.NewAgenticRAGProcessor(config)

    // Advanced query with comprehensive analysis
    request := plugin.AgenticRAGRequest{
        Query: "Analyze the evolution and impact of artificial intelligence technologies",
        Documents: []string{
            `Artificial Intelligence has undergone remarkable evolution since its inception in the 1950s.
             Early AI systems like IBM's Deep Blue demonstrated rule-based approaches to complex problems.
             The field experienced significant breakthroughs with the development of machine learning algorithms,
             particularly neural networks and deep learning architectures. Companies like Google, OpenAI, and
             Anthropic have pioneered large language models that exhibit emergent capabilities. Modern AI systems
             now power applications from autonomous vehicles to medical diagnosis, fundamentally transforming
             industries and society.`,
        },
        Options: plugin.AgenticRAGOptions{
            MaxChunks:              20,
            RecursiveDepth:         3,
            EnableKnowledgeGraph:   true,
            EnableFactVerification: true,
            Temperature:            0.3, // Lower temperature for more focused analysis
        },
    }

    response, err := processor.Process(ctx, request)
    if err != nil {
        log.Fatalf("Processing failed: %v", err)
    }

    // Display comprehensive results
    fmt.Printf("=== AI Evolution Analysis ===\n\n")
    fmt.Printf("Generated Response:\n%s\n\n", response.Answer)

    if response.KnowledgeGraph != nil {
        fmt.Printf("Knowledge Graph Extracted:\n")
        fmt.Printf("- Entities: %d (Organizations, Technologies, Concepts)\n", len(response.KnowledgeGraph.Entities))
        fmt.Printf("- Relations: %d (Development, Usage, Impact relationships)\n", len(response.KnowledgeGraph.Relations))

        // Show key entities
        fmt.Printf("\nKey Entities Identified:\n")
        for _, entity := range response.KnowledgeGraph.Entities[:min(5, len(response.KnowledgeGraph.Entities))] {
            fmt.Printf("- %s (%s) [Confidence: %.2f]\n", entity.Name, entity.Type, entity.Confidence)
        }
    }

    if response.FactVerification != nil {
        fmt.Printf("\nFact Verification:\n")
        fmt.Printf("- Overall Status: %s\n", response.FactVerification.Overall)
        fmt.Printf("- Verified Claims: %d/%d\n",
                   countVerifiedClaims(response.FactVerification.Claims),
                   len(response.FactVerification.Claims))
    }

    fmt.Printf("\nProcessing Metrics:\n")
    fmt.Printf("- Processing Time: %v\n", response.ProcessingMetadata.ProcessingTime)
    fmt.Printf("- LLM Calls Made: %d\n", response.ProcessingMetadata.ModelCalls)
    fmt.Printf("- Tokens Processed: %d\n", response.ProcessingMetadata.TokensUsed)
}

func countVerifiedClaims(claims []plugin.Claim) int {
f claim.Status == "verified" {
            count++
        }
    }
    return count
}

func min(a, b int) int {
    if a < b { return a }
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

This is a **production-ready implementation** that provides:

### ✅ Fully Implemented Features

- **Complete LLM integration** with GenKit Go API v0.6.1
- **Advanced prompt engineering** for all processing stages
- **Real-time relevance scoring** using language models
- **Sophisticated knowledge graph extraction** with entity/relation mapping
- **Comprehensive fact verification** with evidence tracking
- **Robust error handling** with graceful fallbacks
- **Streaming-ready architecture** via GenKit flows
- **Performance optimization** with efficient token usage

### 🚀 Advanced Capabilities

- **Multi-model support**: Works with any GenKit-compatible LLM
- **Configuration flexibility**: Fine-tune every aspect of processing
- **JSON-structured responses**: Reliable parsing of LLM outputs
- **Confidence-based filtering**: Quality control for all extractions
- **Recursive analysis**: Deep document drilling with configurable depth
- **Context preservation**: Maintains document relationships and structure

### 📈 Production Features

- **Comprehensive observability**: Detailed metrics and performance tracking
- **Error resilience**: Automatic fallback mechanisms for reliability
- **Resource optimization**: Efficient token usage and processing
- **Scalable architecture**: Ready for high-volume deployments

## Advanced Examples

For comprehensive usage examples demonstrating all features:

- **[Basic Example](example/main.go)** - Quick start with default configuration
- **[Advanced Example](examples/advanced_agentic_rag/)** - Full-featured implementation with sophisticated analysis

The advanced example showcases:

- Complex technical document analysis
- Knowledge graph construction with confidence thresholds
- Fact verification with evidence tracking
- Performance optimization techniques
- Error handling patterns

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.
