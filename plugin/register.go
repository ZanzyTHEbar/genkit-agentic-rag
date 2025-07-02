// package plugin provides an agentic RAG (Retrieval-Augmented Generation) plugin for Firebase GenKit.
// This package implements the OpenAI Agentic RAG Flow specification with the following key components:
//
// 1. Load entire documents into context window with intelligent context management
// 2. Chunk documents into manageable pieces respecting sentence boundaries
// 3. Prompt model to identify relevant chunks for the query
// 4. Recursively drill down into selected chunks for granular information
// 5. Generate responses based on retrieved information
// 6. Build knowledge graphs from context (if memory is enabled)
// 7. Verify answers for factual accuracy
//
// The implementation follows hexagonal architecture principles and uses functional options pattern
// along with Go best practices for concurrent processing and error handling.
package plugin

import (
	"context"

	"github.com/firebase/genkit/go/genkit"
)

// RegisterPlugin registers the agentic RAG plugin with GenKit
func RegisterPlugin(g *genkit.Genkit, config *AgenticRAGConfig) error {
	if config == nil {
		config = DefaultConfig()
	}

	// Set the GenKit instance in config
	config.Genkit = g

	// If ModelName is set but Model is nil, try to lookup the model
	if config.ModelName != "" && config.Model == nil {
		// The model will be looked up by name when needed
	}

	plugin := NewPlugin(config)
	return plugin.Init(context.Background(), g)
}

// RegisterPluginWithDefaults registers the agentic RAG plugin with default configuration
func RegisterPluginWithDefaults(g *genkit.Genkit) error {
	return RegisterPlugin(g, DefaultConfig())
}
