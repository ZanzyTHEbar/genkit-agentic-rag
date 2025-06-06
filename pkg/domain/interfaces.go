package domain

import (
	"context"
	"io"
)

// VectorStore defines the interface for vector storage operations
type VectorStore interface {
	// Initialize the vector store with the given configuration
	Initialize(ctx context.Context, config map[string]interface{}) error

	// Store documents with their embeddings
	Store(ctx context.Context, documents []Document) error

	// Search for similar documents
	Search(ctx context.Context, query Query) (*QueryResult, error)

	// Delete documents by IDs
	Delete(ctx context.Context, ids []string) error

	// Update a document
	Update(ctx context.Context, document Document) error

	// Get a document by ID
	Get(ctx context.Context, id string) (*Document, error)

	// List all documents with optional filters
	List(ctx context.Context, filters map[string]interface{}, limit, offset int) ([]Document, error)

	// Get collection statistics
	Stats(ctx context.Context) (map[string]interface{}, error)

	// Close the vector store connection
	Close() error
}

// Embedder defines the interface for generating embeddings
type Embedder interface {
	// Generate embeddings for the given texts
	Embed(ctx context.Context, request EmbeddingRequest) (*EmbeddingResponse, error)

	// Get the dimension of embeddings produced by this embedder
	Dimension() int

	// Get the model name
	ModelName() string
}

// Generator defines the interface for text generation
type Generator interface {
	// Generate text based on the request
	Generate(ctx context.Context, request GenerationRequest) (*GenerationResponse, error)

	// Generate text with streaming
	GenerateStream(ctx context.Context, request GenerationRequest, stream chan<- GenerationResponse) error

	// Get available models
	Models() []string
}

// Agent defines the interface for AI agents
type Agent interface {
	// Get the agent configuration
	Config() AgentConfig

	// Process a message and return a response
	Process(ctx context.Context, message string, sessionID string) (*AgentMessage, error)

	// Process with streaming response
	ProcessStream(ctx context.Context, message string, sessionID string, stream chan<- *AgentMessage) error

	// Add a tool to the agent
	AddTool(tool Tool, handler ToolHandler) error

	// Remove a tool from the agent
	RemoveTool(toolName string) error

	// List available tools
	ListTools() []Tool
}

// SessionManager defines the interface for managing conversation sessions
type SessionManager interface {
	// Create a new session
	CreateSession(ctx context.Context, userID, agentID string) (*SessionState, error)

	// Get a session by ID
	GetSession(ctx context.Context, sessionID string) (*SessionState, error)

	// Update a session
	UpdateSession(ctx context.Context, session *SessionState) error

	// Delete a session
	DeleteSession(ctx context.Context, sessionID string) error

	// Add a message to a session
	AddMessage(ctx context.Context, sessionID string, message *AgentMessage) error

	// Get messages from a session with pagination
	GetMessages(ctx context.Context, sessionID string, limit, offset int) ([]AgentMessage, error)

	// Update session context
	UpdateContext(ctx context.Context, sessionID string, context map[string]interface{}) error

	// List sessions for a user
	ListUserSessions(ctx context.Context, userID string, limit, offset int) ([]SessionState, error)
}

// RAGService defines the interface for RAG operations
type RAGService interface {
	// Index documents into the vector store
	IndexDocuments(ctx context.Context, documents []Document, storeName string) error

	// Index a single document
	IndexDocument(ctx context.Context, document Document, storeName string) error

	// Query the RAG system
	Query(ctx context.Context, query string, storeName string, config *RetrievalConfig) (*QueryResult, error)

	// Delete documents from the index
	DeleteDocuments(ctx context.Context, ids []string, storeName string) error

	// Update documents in the index
	UpdateDocuments(ctx context.Context, documents []Document, storeName string) error

	// Get statistics about the index
	GetIndexStats(ctx context.Context, storeName string) (map[string]interface{}, error)
}

// AgentOrchestrator defines the interface for managing multiple agents
type AgentOrchestrator interface {
	// Register a new agent
	RegisterAgent(agent Agent) error

	// Get an agent by ID
	GetAgent(agentID string) (Agent, error)

	// Route a message to the appropriate agent
	RouteMessage(ctx context.Context, message string, sessionID string) (*AgentMessage, error)

	// Create a specialized agent for a specific task
	CreateSpecializedAgent(config AgentConfig) (Agent, error)

	// List all registered agents
	ListAgents() []AgentConfig

	// Remove an agent
	RemoveAgent(agentID string) error
}

// ToolHandler defines the interface for tool execution
type ToolHandler interface {
	// Execute the tool with the given arguments
	Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)

	// Validate the arguments for the tool
	Validate(args map[string]interface{}) error

	// Get the tool definition
	Definition() Tool
}

// DocumentProcessor defines the interface for processing documents before indexing
type DocumentProcessor interface {
	// Process a document (chunking, cleaning, etc.)
	Process(ctx context.Context, document Document) ([]Document, error)

	// Process multiple documents
	ProcessBatch(ctx context.Context, documents []Document) ([]Document, error)

	// Extract text from various file formats
	ExtractText(ctx context.Context, reader io.Reader, contentType string) (string, error)
}

// ConfigManager defines the interface for configuration management
type ConfigManager interface {
	// Load configuration from various sources
	Load(ctx context.Context) error

	// Get configuration value
	Get(key string) interface{}

	// Set configuration value
	Set(key string, value interface{})

	// Get string value with default
	GetString(key, defaultValue string) string

	// Get int value with default
	GetInt(key string, defaultValue int) int

	// Get float value with default
	GetFloat32(key string, defaultValue float32) float32

	// Get bool value with default
	GetBool(key string, defaultValue bool) bool

	// Get map value
	GetStringMap(key string) map[string]interface{}

	// Validate configuration
	Validate() error

	// Watch for configuration changes
	Watch(callback func(key string, value interface{})) error
}

// ErrorHandler defines the interface for error handling
type ErrorHandler interface {
	// Handle an error
	Handle(ctx context.Context, err error) error

	// Wrap an error with additional context
	Wrap(err error, message string, fields map[string]interface{}) error

	// Create a new error
	New(message string, fields map[string]interface{}) error

	// Check if an error is of a specific type
	Is(err error, target error) bool
}

// Metrics defines the interface for collecting metrics
type Metrics interface {
	// Record a counter
	Counter(name string, value int64, tags map[string]string)

	// Record a histogram
	Histogram(name string, value float64, tags map[string]string)

	// Record a gauge
	Gauge(name string, value float64, tags map[string]string)

	// Start timing an operation
	Timer(name string, tags map[string]string) func()
}

// Logger defines the interface for logging
type Logger interface {
	// Log debug message
	Debug(msg string, fields map[string]interface{})

	// Log info message
	Info(msg string, fields map[string]interface{})

	// Log warning message
	Warn(msg string, fields map[string]interface{})

	// Log error message
	Error(msg string, fields map[string]interface{})

	// Log with custom level
	Log(level string, msg string, fields map[string]interface{})

	// Create a child logger with additional fields
	WithFields(fields map[string]interface{}) Logger
}
