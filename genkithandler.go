// Package genkithandler provides Firebase GenKit plugins and utilities.
// This package includes an agentic RAG (Retrieval-Augmented Generation) plugin
// that implements sophisticated document processing and knowledge graph construction.
package genkithandler

import (
	"context"

	"github.com/firebase/genkit/go/genkit"

	"github.com/ZanzyTHEbar/genkithandler/internal/agentic"
)

// InitializeAgenticRAG initializes the agentic RAG plugin with GenKit
func InitializeAgenticRAG(g *genkit.Genkit, config *agentic.AgenticRAGConfig) error {
	return agentic.RegisterPlugin(g, config)
}

// InitializeAgenticRAGWithDefaults initializes the agentic RAG plugin with default configuration
func InitializeAgenticRAGWithDefaults(g *genkit.Genkit) error {
	return agentic.RegisterPluginWithDefaults(g)
}

// NewAgenticRAGProcessor creates a new agentic RAG processor that can be used standalone
func NewAgenticRAGProcessor(config *agentic.AgenticRAGConfig) *agentic.AgenticRAGProcessor {
	return agentic.NewAgenticRAGProcessor(config)
}

// DefaultAgenticRAGConfig returns a default configuration for the agentic RAG system
func DefaultAgenticRAGConfig() *agentic.AgenticRAGConfig {
	return agentic.DefaultConfig()
}

// InitializeAgenticRAGWithPrompts initializes GenKit with prompts directory and the agentic RAG plugin
func InitializeAgenticRAGWithPrompts(promptsDir string, config *agentic.AgenticRAGConfig) (*genkit.Genkit, error) {
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
func InitializeAgenticRAGWithDefaultPrompts(config *agentic.AgenticRAGConfig) (*genkit.Genkit, error) {
	return InitializeAgenticRAGWithPrompts("./prompts", config)
}
