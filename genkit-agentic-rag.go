// package genkit_agentic_rag provides Firebase GenKit plugins and utilities.
// This package includes an agentic RAG (Retrieval-Augmented Generation) plugin
// that implements sophisticated document processing and knowledge graph construction.
package genkit_agentic_rag

import (
	"context"

	"github.com/ZanzyTHEbar/genkit-agentic-rag/plugin"
	"github.com/firebase/genkit/go/genkit"
)

// InitializeAgenticRAG initializes the agentic RAG plugin with GenKit
func InitializeAgenticRAG(g *genkit.Genkit, config *plugin.AgenticRAGConfig) error {
	return plugin.RegisterPlugin(g, config)
}

// InitializeAgenticRAGWithDefaults initializes the agentic RAG plugin with default configuration
func InitializeAgenticRAGWithDefaults(g *genkit.Genkit) error {
	return plugin.RegisterPluginWithDefaults(g)
}

// NewAgenticRAGProcessor creates a new agentic RAG processor that can be used standalone
func NewAgenticRAGProcessor(config *plugin.AgenticRAGConfig) *plugin.AgenticRAGProcessor {
	return plugin.NewAgenticRAGProcessor(config)
}

// DefaultAgenticRAGConfig returns a default configuration for the agentic RAG system
func DefaultAgenticRAGConfig() *plugin.AgenticRAGConfig {
	return plugin.DefaultConfig()
}

// InitializeAgenticRAGWithPrompts initializes GenKit with prompts directory and the agentic RAG plugin
func InitializeAgenticRAGWithPrompts(promptsDir string, config *plugin.AgenticRAGConfig) (*genkit.Genkit, error) {
	// Initialize GenKit with prompts directory
	g, err := genkit.Init(
		context.Background(),
		genkit.WithPromptDir(promptsDir),
	)
	if err != nil {
		return nil, err
	}

	// Configure prompts directory in config
	if config != nil {
		config.Prompts.Directory = promptsDir
	}

	// Initialize the agentic RAG plugin
	if err := InitializeAgenticRAG(g, config); err != nil {
		return nil, err
	}

	return g, nil
}

// InitializeAgenticRAGWithDefaultPrompts initializes with default prompts directory ("./prompts")
func InitializeAgenticRAGWithDefaultPrompts(config *plugin.AgenticRAGConfig) (*genkit.Genkit, error) {
	return InitializeAgenticRAGWithPrompts("./prompts", config)
}
