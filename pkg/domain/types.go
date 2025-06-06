package domain

import (
	"time"

	"github.com/google/uuid"
)

// Document represents a document with its content and metadata for RAG operations
type Document struct {
	ID          string                 `json:"id"`
	Content     string                 `json:"content"`
	Metadata    map[string]interface{} `json:"metadata"`
	Embedding   []float32              `json:"embedding,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Source      string                 `json:"source,omitempty"`
	ChunkIndex  int                    `json:"chunk_index,omitempty"`
	TotalChunks int                    `json:"total_chunks,omitempty"`
}

// Query represents a search query with options
type Query struct {
	Text        string                 `json:"text"`
	Embedding   []float32              `json:"embedding,omitempty"`
	Filters     map[string]interface{} `json:"filters,omitempty"`
	Limit       int                    `json:"limit,omitempty"`
	Threshold   float32                `json:"threshold,omitempty"`
	IncludeText bool                   `json:"include_text"`
}

// QueryResult represents the result of a query operation
type QueryResult struct {
	Documents []Document `json:"documents"`
	Scores    []float32  `json:"scores,omitempty"`
	Total     int        `json:"total"`
	Query     Query      `json:"query"`
}

// AgentMessage represents a message in an agent conversation
type AgentMessage struct {
	ID        string                 `json:"id"`
	Role      string                 `json:"role"` // "user", "assistant", "system", "tool"
	Content   string                 `json:"content"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	ToolCalls []ToolCall             `json:"tool_calls,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// ToolCall represents a tool invocation
type ToolCall struct {
	ID       string                 `json:"id"`
	Name     string                 `json:"name"`
	Args     map[string]interface{} `json:"args"`
	Result   interface{}            `json:"result,omitempty"`
	Error    string                 `json:"error,omitempty"`
	Duration time.Duration          `json:"duration,omitempty"`
}

// SessionState represents the state of a conversation session
type SessionState struct {
	ID          string                 `json:"id"`
	UserID      string                 `json:"user_id,omitempty"`
	AgentID     string                 `json:"agent_id"`
	Messages    []AgentMessage         `json:"messages"`
	Context     map[string]interface{} `json:"context"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	ActiveTools []string               `json:"active_tools,omitempty"`
}

// AgentConfig represents the configuration for an agent
type AgentConfig struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	SystemPrompt    string                 `json:"system_prompt"`
	ModelName       string                 `json:"model_name"`
	Temperature     float32                `json:"temperature,omitempty"`
	MaxTokens       int                    `json:"max_tokens,omitempty"`
	Tools           []string               `json:"tools"`
	VectorStoreName string                 `json:"vector_store_name,omitempty"`
	RetrievalConfig *RetrievalConfig       `json:"retrieval_config,omitempty"`
	AgentType       AgentType              `json:"agent_type"`
	Specializations []string               `json:"specializations,omitempty"`
	ParentAgentID   string                 `json:"parent_agent_id,omitempty"`
	ChildAgentIDs   []string               `json:"child_agent_ids,omitempty"`
	Config          map[string]interface{} `json:"config,omitempty"`
}

// RetrievalConfig represents configuration for RAG retrieval
type RetrievalConfig struct {
	TopK                int     `json:"top_k"`
	SimilarityThreshold float32 `json:"similarity_threshold"`
	MaxDocumentLength   int     `json:"max_document_length"`
	OverlapPercentage   float32 `json:"overlap_percentage"`
	RetrievalStrategy   string  `json:"retrieval_strategy"` // "simple", "multi_query", "hyde"
}

// AgentType represents the type of agent
type AgentType string

const (
	AgentTypeOrchestrator AgentType = "orchestrator"
	AgentTypeSpecialist   AgentType = "specialist"
	AgentTypeRetrieval    AgentType = "retrieval"
	AgentTypeTool         AgentType = "tool"
)

// EmbeddingRequest represents a request to generate embeddings
type EmbeddingRequest struct {
	Texts     []string `json:"texts"`
	ModelName string   `json:"model_name,omitempty"`
}

// EmbeddingResponse represents the response from embedding generation
type EmbeddingResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	ModelName  string      `json:"model_name"`
	Usage      Usage       `json:"usage,omitempty"`
}

// Usage represents token/request usage information
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// GenerationRequest represents a request for text generation
type GenerationRequest struct {
	Messages    []AgentMessage         `json:"messages"`
	ModelName   string                 `json:"model_name,omitempty"`
	Temperature float32                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Tools       []Tool                 `json:"tools,omitempty"`
	Context     map[string]interface{} `json:"context,omitempty"`
	Stream      bool                   `json:"stream,omitempty"`
}

// GenerationResponse represents the response from text generation
type GenerationResponse struct {
	Message   AgentMessage `json:"message"`
	Usage     Usage        `json:"usage"`
	ModelName string       `json:"model_name"`
	Finished  bool         `json:"finished"`
}

// Tool represents a tool that can be called by agents
type Tool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Required    []string               `json:"required,omitempty"`
}

// NewDocument creates a new document with generated ID and timestamps
func NewDocument(content string, metadata map[string]interface{}) *Document {
	now := time.Now()
	return &Document{
		ID:        uuid.New().String(),
		Content:   content,
		Metadata:  metadata,
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// NewSession creates a new session with generated ID and timestamps
func NewSession(userID, agentID string) *SessionState {
	now := time.Now()
	return &SessionState{
		ID:        uuid.New().String(),
		UserID:    userID,
		AgentID:   agentID,
		Messages:  make([]AgentMessage, 0),
		Context:   make(map[string]interface{}),
		CreatedAt: now,
		UpdatedAt: now,
	}
}

// NewAgentMessage creates a new agent message with generated ID and timestamp
func NewAgentMessage(role, content string) *AgentMessage {
	return &AgentMessage{
		ID:        uuid.New().String(),
		Role:      role,
		Content:   content,
		Metadata:  make(map[string]interface{}),
		ToolCalls: make([]ToolCall, 0),
		Timestamp: time.Now(),
	}
}
