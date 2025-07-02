// Package genkithandler provides Firebase GenKit plugins and utilities.
// This package includes an agentic RAG (Retrieval-Augmented Generation) plugin
// that implements sophisticated document processing and knowledge graph construction.
package genkithandler

import (
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
