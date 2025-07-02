package internal

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

const PluginID = "agentic-rag"

// AgenticRAGPlugin represents the GenKit plugin for agentic RAG
type AgenticRAGPlugin struct {
	processor *AgenticRAGProcessor
	config    *AgenticRAGConfig
}

// NewPlugin creates a new agentic RAG plugin
func NewPlugin(config *AgenticRAGConfig) *AgenticRAGPlugin {
	if config == nil {
		config = DefaultConfig()
	}

	return &AgenticRAGPlugin{
		processor: NewAgenticRAGProcessor(config),
		config:    config,
	}
}

// Name returns the plugin name
func (p *AgenticRAGPlugin) Name() string {
	return PluginID
}

// Init initializes the plugin with GenKit
func (p *AgenticRAGPlugin) Init(ctx context.Context, g *genkit.Genkit) error {
	// Store GenKit instance in config for processor access
	p.config.Genkit = g

	// Initialize prompts and custom helpers
	if err := p.processor.initializePrompts(ctx); err != nil {
		return fmt.Errorf("failed to initialize prompts: %w", err)
	}

	// Register the main agentic RAG flow
	if err := p.registerFlows(ctx, g); err != nil {
		return fmt.Errorf("failed to register flows: %w", err)
	}

	// Register tools for document processing
	if err := p.registerTools(ctx, g); err != nil {
		return fmt.Errorf("failed to register tools: %w", err)
	}

	return nil
}

// registerFlows registers the agentic RAG flows
func (p *AgenticRAGPlugin) registerFlows(ctx context.Context, g *genkit.Genkit) error {
	// Main agentic RAG streaming flow using correct GenKit Go API
	genkit.DefineStreamingFlow(
		g,
		"agenticRAG",
		func(ctx context.Context, input AgenticRAGRequest, cb func(context.Context, *AgenticRAGResponse) error) (*AgenticRAGResponse, error) {
			// Use the processor to handle the full agentic RAG pipeline
			response, err := p.processor.Process(ctx, input)
			if err != nil {
				return nil, err
			}

			// If streaming callback is provided, stream the response
			if cb != nil {
				if err := cb(ctx, response); err != nil {
					return nil, err
				}
			}

			return response, nil
		},
	)

	// Also register a simple non-streaming flow for basic usage
	genkit.DefineFlow(g, "agenticRAGSimple", func(ctx context.Context, input AgenticRAGRequest) (*AgenticRAGResponse, error) {
		return p.processor.Process(ctx, input)
	})

	return nil
}

// registerTools registers helper tools
func (p *AgenticRAGPlugin) registerTools(ctx context.Context, g *genkit.Genkit) error {
	// Document chunking tool
	genkit.DefineTool(
		g,
		"chunkDocument",
		"Chunks a document into smaller pieces respecting sentence boundaries",
		func(ctx *ai.ToolContext, input ChunkDocumentRequest) (ChunkDocumentResponse, error) {
			doc := Document{
				ID:      "temp_doc",
				Content: input.Content,
				Source:  "user_input",
			}

			chunks, err := p.processor.chunkDocument(ctx, doc, input.MaxChunks)
			if err != nil {
				return ChunkDocumentResponse{}, err
			}

			return ChunkDocumentResponse{
				Chunks:      chunks,
				ChunkCount:  len(chunks),
				ProcessedAt: "now", // Simplified for MVP
			}, nil
		},
	)

	// Relevance scoring tool
	genkit.DefineTool(
		g,
		"scoreRelevance",
		"Scores the relevance of text chunks against a query",
		func(ctx *ai.ToolContext, input RelevanceScoreRequest) (RelevanceScoreResponse, error) {
			scores := make([]RelevanceScore, len(input.Chunks))

			for i, chunkText := range input.Chunks {
				score := p.processor.calculateRelevanceScore(input.Query, chunkText)
				scores[i] = RelevanceScore{
					ChunkIndex: i,
					Score:      score,
					ChunkText:  chunkText,
				}
			}

			return RelevanceScoreResponse{
				Scores: scores,
			}, nil
		},
	)

	// Knowledge graph extraction tool
	if p.config.KnowledgeGraph.Enabled {
		genkit.DefineTool(
			g,
			"extractKnowledgeGraph",
			"Extracts entities and relations to build a knowledge graph",
			func(ctx *ai.ToolContext, input KnowledgeGraphRequest) (KnowledgeGraphResponse, error) {
				// Convert input chunks to DocumentChunk format
				chunks := make([]DocumentChunk, len(input.Chunks))
				for i, chunkText := range input.Chunks {
					chunks[i] = DocumentChunk{
						ID:      fmt.Sprintf("chunk_%d", i),
						Content: chunkText,
					}
				}

				kg, err := p.processor.buildKnowledgeGraph(ctx, chunks)
				if err != nil {
					return KnowledgeGraphResponse{}, err
				}

				return KnowledgeGraphResponse{
					KnowledgeGraph: kg,
				}, nil
			},
		)
	}

	return nil
}
