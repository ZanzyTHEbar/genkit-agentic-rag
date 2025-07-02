package plugin

import (
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// Core request/response types for agentic RAG flow

// AgenticRAGRequest represents a request for the agentic RAG flow
type AgenticRAGRequest struct {
	Query     string            `json:"query" jsonschema_description:"The user's query or question"`
	Documents []string          `json:"documents,omitempty" jsonschema_description:"Documents to process (URLs, file paths, or raw text)"`
	Options   AgenticRAGOptions `json:"options,omitempty" jsonschema_description:"Processing options"`
}

// AgenticRAGOptions contains processing options
type AgenticRAGOptions struct {
	MaxChunks              int     `json:"max_chunks,omitempty" jsonschema_description:"Maximum number of chunks to process (default: 20)"`
	RecursiveDepth         int     `json:"recursive_depth,omitempty" jsonschema_description:"Maximum recursive processing depth (default: 3)"`
	EnableKnowledgeGraph   bool    `json:"enable_knowledge_graph,omitempty" jsonschema_description:"Whether to build knowledge graph"`
	EnableFactVerification bool    `json:"enable_fact_verification,omitempty" jsonschema_description:"Whether to verify facts in response"`
	Temperature            float32 `json:"temperature,omitempty" jsonschema_description:"Temperature for generation (default: 0.7)"`
}

// AgenticRAGResponse represents the response from agentic RAG flow
type AgenticRAGResponse struct {
	Answer             string             `json:"answer" jsonschema_description:"The generated answer"`
	RelevantChunks     []ProcessedChunk   `json:"relevant_chunks" jsonschema_description:"Chunks used to generate answer"`
	KnowledgeGraph     *KnowledgeGraph    `json:"knowledge_graph,omitempty" jsonschema_description:"Knowledge graph if enabled"`
	FactVerification   *FactVerification  `json:"fact_verification,omitempty" jsonschema_description:"Fact verification results if enabled"`
	ProcessingMetadata ProcessingMetadata `json:"processing_metadata" jsonschema_description:"Processing metadata"`
}

// Document represents a document to be processed
type Document struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Source   string                 `json:"source"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// DocumentChunk represents a chunk of a document
type DocumentChunk struct {
	ID             string  `json:"id"`
	Content        string  `json:"content"`
	DocumentID     string  `json:"document_id"`
	ChunkIndex     int     `json:"chunk_index"`
	StartIndex     int     `json:"start_index"`
	EndIndex       int     `json:"end_index"`
	RelevanceScore float64 `json:"relevance_score,omitempty"`
}

// ProcessedChunk represents a chunk that has been processed and scored
type ProcessedChunk struct {
	Chunk     DocumentChunk          `json:"chunk"`
	Entities  []Entity               `json:"entities,omitempty"`
	Relations []Relation             `json:"relations,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Entity represents an extracted entity
type Entity struct {
	ID         string                 `json:"id"`
	Name       string                 `json:"name"`
	Type       string                 `json:"type"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Confidence float64                `json:"confidence"`
}

// Relation represents a relationship between entities
type Relation struct {
	ID         string                 `json:"id"`
	Subject    string                 `json:"subject"`
	Predicate  string                 `json:"predicate"`
	Object     string                 `json:"object"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	Confidence float64                `json:"confidence"`
}

// KnowledgeGraph represents the constructed knowledge graph
type KnowledgeGraph struct {
	Entities  []Entity               `json:"entities"`
	Relations []Relation             `json:"relations"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// FactVerification represents fact verification results
type FactVerification struct {
	Claims   []Claim                `json:"claims"`
	Overall  string                 `json:"overall"` // "verified", "partially_verified", "unverified"
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Claim represents a factual claim and its verification
type Claim struct {
	Text       string   `json:"text"`
	Status     string   `json:"status"` // "verified", "refuted", "inconclusive"
	Confidence float64  `json:"confidence"`
	Evidence   []string `json:"evidence,omitempty"`
}

// ProcessingMetadata contains metadata about the processing
type ProcessingMetadata struct {
	ProcessingTime  time.Duration `json:"processing_time"`
	ChunksProcessed int           `json:"chunks_processed"`
	RecursiveLevels int           `json:"recursive_levels"`
	ModelCalls      int           `json:"model_calls"`
	TokensUsed      int           `json:"tokens_used"`
}

// AgenticRAGConfig contains configuration for the agentic RAG system
type AgenticRAGConfig struct {
	Genkit           *genkit.Genkit         `json:"-"`          // GenKit instance (not serialized)
	Model            ai.Model               `json:"-"`          // Model instance (not serialized)
	ModelName        string                 `json:"model_name"` // Model name for serialization
	Processing       ProcessingConfig       `json:"processing"`
	KnowledgeGraph   KnowledgeGraphConfig   `json:"knowledge_graph"`
	FactVerification FactVerificationConfig `json:"fact_verification"`
	Prompts          PromptsConfig          `json:"prompts"`
}

// ModelConfig contains model configuration
type ModelConfig struct {
	Provider    string  `json:"provider"`
	Model       string  `json:"model"`
	APIKey      string  `json:"api_key"`
	Temperature float32 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

// ProcessingConfig contains processing configuration
type ProcessingConfig struct {
	DefaultChunkSize      int  `json:"default_chunk_size"`
	DefaultMaxChunks      int  `json:"default_max_chunks"`
	DefaultRecursiveDepth int  `json:"default_recursive_depth"`
	RespectSentences      bool `json:"respect_sentences"`
}

// KnowledgeGraphConfig contains knowledge graph configuration
type KnowledgeGraphConfig struct {
	Enabled                bool     `json:"enabled"`
	EntityTypes            []string `json:"entity_types"`
	RelationTypes          []string `json:"relation_types"`
	MinConfidenceThreshold float64  `json:"min_confidence_threshold"`
}

// FactVerificationConfig contains fact verification configuration
type FactVerificationConfig struct {
	Enabled            bool    `json:"enabled"`
	RequireEvidence    bool    `json:"require_evidence"`
	MinConfidenceScore float64 `json:"min_confidence_score"`
}

// PromptsConfig contains prompt configuration
type PromptsConfig struct {
	Directory                 string            `json:"directory"`                   // Directory containing .prompt files
	RelevanceScoringPrompt    string            `json:"relevance_scoring_prompt"`    // Name of relevance scoring prompt
	ResponseGenerationPrompt  string            `json:"response_generation_prompt"`  // Name of response generation prompt
	KnowledgeExtractionPrompt string            `json:"knowledge_extraction_prompt"` // Name of knowledge extraction prompt
	FactVerificationPrompt    string            `json:"fact_verification_prompt"`    // Name of fact verification prompt
	Variants                  map[string]string `json:"variants,omitempty"`          // Prompt variants for A/B testing
	CustomHelpers             bool              `json:"custom_helpers"`              // Whether to register custom helpers
}

// Tool request/response types

// ChunkDocumentRequest represents a request to chunk a document
type ChunkDocumentRequest struct {
	Content   string `json:"content" jsonschema_description:"Document content to chunk"`
	MaxChunks int    `json:"max_chunks,omitempty" jsonschema_description:"Maximum number of chunks to create"`
}

// ChunkDocumentResponse represents the response from document chunking
type ChunkDocumentResponse struct {
	Chunks      []DocumentChunk `json:"chunks" jsonschema_description:"Created chunks"`
	ChunkCount  int             `json:"chunk_count" jsonschema_description:"Number of chunks created"`
	ProcessedAt string          `json:"processed_at" jsonschema_description:"Processing timestamp"`
}

// RelevanceScoreRequest represents a request to score chunk relevance
type RelevanceScoreRequest struct {
	Query  string   `json:"query" jsonschema_description:"Query to score against"`
	Chunks []string `json:"chunks" jsonschema_description:"Text chunks to score"`
}

// RelevanceScoreResponse represents the response from relevance scoring
type RelevanceScoreResponse struct {
	Scores []RelevanceScore `json:"scores" jsonschema_description:"Relevance scores for each chunk"`
}

// RelevanceScore represents the relevance score for a single chunk
type RelevanceScore struct {
	ChunkIndex int     `json:"chunk_index" jsonschema_description:"Index of the chunk"`
	Score      float64 `json:"score" jsonschema_description:"Relevance score (0-1)"`
	ChunkText  string  `json:"chunk_text" jsonschema_description:"Text content of the chunk"`
}

// KnowledgeGraphRequest represents a request to extract knowledge graph
type KnowledgeGraphRequest struct {
	Chunks []string `json:"chunks" jsonschema_description:"Text chunks to process"`
}

// KnowledgeGraphResponse represents the response from knowledge graph extraction
type KnowledgeGraphResponse struct {
	KnowledgeGraph *KnowledgeGraph `json:"knowledge_graph" jsonschema_description:"Extracted knowledge graph"`
}
