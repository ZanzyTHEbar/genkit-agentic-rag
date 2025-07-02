# Advanced Agentic RAG Example

This example demonstrates the enhanced Agentic RAG pipeline with proper GenKit Go API integration.

## Features Implemented

### 🚀 Core Agentic RAG Flow

- **LLM-based chunk relevance scoring**: Uses actual LLM calls to intelligently score document chunks
- **Advanced response generation**: Sophisticated prompt engineering for comprehensive, cited responses
- **Recursive drilling**: Deep analysis capability with configurable depth

### 🧠 Knowledge Graph Construction

- **Entity extraction**: Identifies PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, TECHNOLOGY
- **Relationship mapping**: Discovers WORKS_FOR, LOCATED_IN, FOUNDED, DEVELOPED, etc.
- **Confidence-based filtering**: Only includes high-confidence entities and relations

### ✅ Fact Verification

- **Claim decomposition**: Breaks responses into verifiable claims
- **Source verification**: Checks each claim against source documents
- **Evidence tracking**: Provides specific evidence for each verified claim

### 🔧 GenKit Integration

- **Proper model handling**: Supports both model instances and model names
- **Streaming support**: Ready for streaming responses (via plugin flows)
- **Configuration flexibility**: Comprehensive configuration options
- **Error handling**: Robust fallback mechanisms

## Usage

### Basic Setup

```go
import (
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/googlegenai"
    "github.com/ZanzyTHEbar/genkit-agentic-rag"
    "github.com/ZanzyTHEbar/genkit-agentic-rag/internal"
)

// Initialize GenKit
g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

// Configure Agentic RAG
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
        EntityTypes:            []string{"PERSON", "ORGANIZATION", "LOCATION", "CONCEPT"},
        RelationTypes:          []string{"WORKS_FOR", "LOCATED_IN", "FOUNDED"},
        MinConfidenceThreshold: 0.8,
    },
}

// Initialize plugin
err = agentic-rag.InitializeAgenticRAG(g, config)
```

### Advanced Processing

```go
// Create processor
processor := agentic-rag.NewAgenticRAGProcessor(config)

// Process with full features
request := plugin.AgenticRAGRequest{
    Query: "Your question here",
    Documents: []string{"Document content..."},
    Options: plugin.AgenticRAGOptions{
        MaxChunks:              20,
        RecursiveDepth:         3,
        EnableKnowledgeGraph:   true,
        EnableFactVerification: true,
        Temperature:            0.3,
    },
}

response, err := processor.Process(ctx, request)
```

## API Improvements

### LLM-Powered Relevance Scoring

Instead of simple keyword matching:

```go
// OLD: Simple keyword matching
score := calculateKeywordScore(query, chunk)

// NEW: LLM-based relevance scoring
prompt := fmt.Sprintf(`Score relevance of chunks for query: %s...`, query)
response, err := genkit.Generate(ctx, g, ai.WithPrompt(prompt))
```

### Sophisticated Response Generation

Enhanced prompt engineering:

```go
prompt := fmt.Sprintf(`You are an expert AI assistant...
Context Information:
%s

Instructions:
1. Answer using ONLY the provided context
2. Be comprehensive but concise
3. Cite sources (e.g., "According to Source 1...")
4. State limitations if context is insufficient

Answer:`, context)
```

### Knowledge Graph Extraction

Real entity and relation extraction:

```go
prompt := fmt.Sprintf(`Extract entities and relationships...
ENTITIES (types: %s):
- Identify important entities with confidence scores
RELATIONS (types: %s):
- Map relationships between entities

Respond with JSON...`, entityTypes, relationTypes)
```

## Configuration Options

### Processing Configuration

- `DefaultChunkSize`: Optimal chunk size for analysis
- `DefaultMaxChunks`: Maximum chunks to process
- `DefaultRecursiveDepth`: How deep to drill down
- `RespectSentences`: Maintain sentence boundaries

### Knowledge Graph Configuration

- `Enabled`: Toggle knowledge graph construction
- `EntityTypes`: Types of entities to extract
- `RelationTypes`: Types of relationships to identify
- `MinConfidenceThreshold`: Minimum confidence for inclusion

### Model Configuration

- `Genkit`: GenKit instance
- `Model`: Specific model instance (optional)
- `ModelName`: Model name for lookup

## Response Structure

```go
type AgenticRAGResponse struct {
    Answer             string             // Generated answer
    RelevantChunks     []ProcessedChunk   // Chunks used
    KnowledgeGraph     *KnowledgeGraph    // Extracted graph
    FactVerification   *FactVerification  // Verification results
    ProcessingMetadata ProcessingMetadata // Performance metrics
}
```

## Error Handling

The implementation includes robust error handling:

- **LLM failures**: Automatic fallback to simpler methods
- **JSON parsing errors**: Graceful degradation
- **Network issues**: Retry logic where appropriate
- **Configuration errors**: Clear error messages

## Performance Considerations

- **Token optimization**: Efficient prompt design
- **Parallel processing**: Where possible (future enhancement)
- **Caching**: Model responses can be cached
- **Streaming**: Ready for streaming implementations

## Running the Example

```bash
cd examples/advanced_agentic_rag
export GOOGLE_GENAI_API_KEY="your-api-key"
go run main.go
```

## Next Steps

This implementation provides a solid foundation for:

1. **Production deployment**: Ready for real-world usage
2. **Custom models**: Easy to swap different LLMs
3. **Streaming responses**: Framework supports streaming
4. **Advanced RAG**: Foundation for more sophisticated patterns
5. **Integration**: Can be embedded in larger applications

The enhanced Agentic RAG pipeline now properly leverages GenKit's capabilities while maintaining the sophisticated multi-step reasoning process outlined in the original specification.
