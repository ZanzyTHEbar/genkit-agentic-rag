package types

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// Core Genkit Integration Types

// StructuredResponse represents a structured AI response with schema validation
type StructuredResponse struct {
	Data      interface{} `json:"data"`
	Schema    string      `json:"schema"`
	Metadata  Metadata    `json:"metadata"`
	RequestID string      `json:"request_id"`
}

// Metadata contains response metadata and context
type Metadata struct {
	Model          string            `json:"model"`
	Provider       string            `json:"provider"`
	TokensUsed     int               `json:"tokens_used,omitempty"`
	ProcessingTime time.Duration     `json:"processing_time"`
	Confidence     float64           `json:"confidence,omitempty"`
	Context        map[string]string `json:"context,omitempty"`
	Timestamp      time.Time         `json:"timestamp"`
}

// GenerateRequest represents a request to generate AI content
type GenerateRequest struct {
	Context     context.Context        `json:"-"`
	Prompt      string                 `json:"prompt"`
	Model       string                 `json:"model,omitempty"`
	Provider    string                 `json:"provider,omitempty"`
	Temperature float64                `json:"temperature,omitempty"`
	MaxTokens   int                    `json:"max_tokens,omitempty"`
	Tools       []ToolDefinition       `json:"tools,omitempty"`
	Schema      string                 `json:"schema,omitempty"`
	Context_    map[string]interface{} `json:"context,omitempty"`
	RequestID   string                 `json:"request_id"`
}

// GenerateResponse represents the response from AI generation
type GenerateResponse struct {
	Content        string      `json:"content"`
	ToolCalls      []ToolCall  `json:"tool_calls,omitempty"`
	FinishReason   string      `json:"finish_reason"`
	Usage          TokenUsage  `json:"usage"`
	Metadata       Metadata    `json:"metadata"`
	StructuredData interface{} `json:"structured_data,omitempty"`
}

// TokenUsage represents token usage information
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// StreamChunk represents a streaming response chunk
type StreamChunk struct {
	Content      string    `json:"content,omitempty"`
	Delta        string    `json:"delta,omitempty"`
	ToolCall     *ToolCall `json:"tool_call,omitempty"`
	FinishReason string    `json:"finish_reason,omitempty"`
	Error        error     `json:"error,omitempty"`
	Done         bool      `json:"done"`
}

// ToolCall represents a tool function call from the AI
type ToolCall struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Arguments map[string]interface{} `json:"arguments"`
}

// ToolResult represents the result of a tool execution
type ToolResult struct {
	ToolCallID string      `json:"tool_call_id"`
	Output     interface{} `json:"output"`
	Error      error       `json:"error,omitempty"`
	Success    bool        `json:"success"`
}

// ToolRequest represents a request to execute a tool
type ToolRequest struct {
	Context   context.Context `json:"-"`
	ToolCall  ToolCall        `json:"tool_call"`
	RequestID string          `json:"request_id"`
}

// ToolResponse represents the response from tool execution
type ToolResponse struct {
	Result    ToolResult `json:"result"`
	Metadata  Metadata   `json:"metadata"`
	RequestID string     `json:"request_id"`
}

// ToolDefinition defines a tool that can be called by the AI
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Handler     ToolHandler            `json:"-"`
}

// ToolHandler is the function signature for tool implementations
type ToolHandler func(ctx context.Context, args map[string]interface{}) (interface{}, error)

// AIProvider interface defines the contract for AI providers
type AIProvider interface {
	// GenerateResponse generates a single response
	GenerateResponse(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error)

	// GenerateStream generates a streaming response
	GenerateStream(ctx context.Context, req *GenerateRequest) (<-chan StreamChunk, error)

	// CallTool executes a tool call
	CallTool(ctx context.Context, req *ToolRequest) (*ToolResponse, error)

	// SupportedModels returns the list of supported models
	SupportedModels() []string

	// Name returns the provider name
	Name() string

	// Initialize sets up the provider with configuration
	Initialize(config map[string]interface{}) error
}

// BackupToolInput defines the input for the backup tool.
// For now, it's empty, implying a full backup. It can be extended with options.
type BackupToolInput struct {
	// Example: BackupTarget string (e.g., "deskfs_only", "centraldb_only", "all")
}

// BackupToolOutput defines the output structure for the backup tool.
// This is what an LLM would receive as a structured response if it invoked this tool.
type BackupToolOutput struct {
	DeskFSBackupPath    string `json:"deskFSBackupPath,omitempty"`
	CentralDBBackupPath string `json:"centralDBBackupPath,omitempty"`
	Message             string `json:"message,omitempty"`
}

func (b BackupToolOutput) Error() string {
	// Only return a non-empty string if the message indicates an error or failure
	if b.Message == "" ||
		(b.Message != "" && !(containsSuccess(b.Message))) {
		return fmt.Sprintf("DeskFSBackupPath: %s, CentralDBBackupPath: %s, Message: %s", b.DeskFSBackupPath, b.CentralDBBackupPath, b.Message)
	}
	return ""
}

// containsSuccess checks if the message indicates a successful backup
func containsSuccess(msg string) bool {
	return contains(msg, "success") || contains(msg, "succeed")
}

// contains is a helper for case-insensitive substring search
func contains(s, substr string) bool {
	return len(s) >= len(substr) && // quick check
		(len(s) > 0 && len(substr) > 0 &&
			(stringContainsFold(s, substr)))
}

// stringContainsFold is a case-insensitive substring search
func stringContainsFold(s, substr string) bool {
	s, substr = toLower(s), toLower(substr)
	return len(substr) > 0 && len(s) > 0 && (indexOf(s, substr) >= 0)
}

// toLower returns a lower-case version of the string
func toLower(s string) string {
	b := []byte(s)
	for i := 0; i < len(b); i++ {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 'a' - 'A'
		}
	}
	return string(b)
}

// indexOf returns the index of substr in s, or -1 if not found
func indexOf(s, substr string) int {
	n, m := len(s), len(substr)
	for i := 0; i <= n-m; i++ {
		if s[i:i+m] == substr {
			return i
		}
	}
	return -1
}

// SessionID generates a unique session identifier.
func SessionID() string {
	return uuid.New().String()
}
