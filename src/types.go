// Package genkithandler provides integration with the genkit AI platform.
package genkithandler

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

// Enhanced Error Types

// ErrorCode represents specific error types
type ErrorCode string

const (
	ErrorCodeInvalidInput           ErrorCode = "INVALID_INPUT"
	ErrorCodeProviderError          ErrorCode = "PROVIDER_ERROR"
	ErrorCodeToolError              ErrorCode = "TOOL_ERROR"
	ErrorCodeValidationError        ErrorCode = "VALIDATION_ERROR"
	ErrorCodeAuthError              ErrorCode = "AUTH_ERROR"
	ErrorCodeRateLimit              ErrorCode = "RATE_LIMIT"
	ErrorCodeTimeout                ErrorCode = "TIMEOUT"
	ErrorCodeInternalError          ErrorCode = "INTERNAL_ERROR"
	ErrorCodeSchemaError            ErrorCode = "SCHEMA_ERROR"
	ErrorCodeContextError           ErrorCode = "CONTEXT_ERROR"
	ErrorCodeToolAlreadyExists      ErrorCode = "TOOL_ALREADY_EXISTS"
	ErrorCodeToolExecutionFailed    ErrorCode = "TOOL_EXECUTION_FAILED"
	ErrorCodeToolNotFound           ErrorCode = "TOOL_NOT_FOUND"
	ErrorCodeSerialization          ErrorCode = "SERIALIZATION_ERROR"
	ErrorCodeTypeConversion         ErrorCode = "TYPE_CONVERSION_ERROR"
	ErrorCodeDependencyError        ErrorCode = "DEPENDENCY_ERROR"
	ErrorCodeBackupFailed           ErrorCode = "BACKUP_FAILED"
	ErrorCodeToolRegistrationFailed ErrorCode = "TOOL_REGISTRATION_FAILED"
)

// GenkitError represents an enhanced error with context
type GenkitError struct {
	Code      ErrorCode              `json:"code"`
	Message   string                 `json:"message"`
	Context   map[string]interface{} `json:"context,omitempty"`
	Provider  string                 `json:"provider,omitempty"`
	RequestID string                 `json:"request_id,omitempty"`
	Retryable bool                   `json:"retryable"`
	Timestamp time.Time              `json:"timestamp"`
	Cause     error                  `json:"cause,omitempty"`
}

func (e *GenkitError) Error() string {
	return fmt.Sprintf("[%s] %s (Provider: %s, RequestID: %s)", e.Code, e.Message, e.Provider, e.RequestID)
}

func (e *GenkitError) Unwrap() error {
	return e.Cause
}

// NewGenkitError creates a new GenkitError
func NewGenkitError(code ErrorCode, message string) *GenkitError {
	return &GenkitError{
		Code:      code,
		Message:   message,
		Context:   make(map[string]interface{}),
		Retryable: isRetryableError(code),
		Timestamp: time.Now(),
	}
}

// WithContext adds context to the error
func (e *GenkitError) WithContext(key string, value interface{}) *GenkitError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// WithProvider sets the provider information
func (e *GenkitError) WithProvider(provider string) *GenkitError {
	e.Provider = provider
	return e
}

// WithRequestID sets the request ID
func (e *GenkitError) WithRequestID(requestID string) *GenkitError {
	e.RequestID = requestID
	return e
}

// WithCause sets the underlying cause
func (e *GenkitError) WithCause(cause error) *GenkitError {
	e.Cause = cause
	return e
}

// isRetryableError determines if an error type is retryable
func isRetryableError(code ErrorCode) bool {
	switch code {
	case ErrorCodeRateLimit, ErrorCodeTimeout, ErrorCodeProviderError:
		return true
	default:
		return false
	}
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

// AI-powered file organization types

// FileOrganizationResult represents the result of AI-powered file organization
type FileOrganizationResult struct {
	SuggestedActions []OrganizationAction `json:"suggested_actions"`
	Confidence       float64              `json:"confidence"`
	Reasoning        string               `json:"reasoning"`
}

// OrganizationAction represents a specific file organization action
type OrganizationAction struct {
	Action     string   `json:"action"` // "move", "rename", "create_folder", "tag"
	FileName   string   `json:"file_name"`
	SourcePath string   `json:"source_path"`
	TargetPath string   `json:"target_path"`
	NewName    string   `json:"new_name,omitempty"`
	FolderName string   `json:"folder_name,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	Confidence float64  `json:"confidence"`
	Reasoning  string   `json:"reasoning"`
}

// FileCategorization represents the result of AI-powered file categorization
type FileCategorization struct {
	Category    string   `json:"category"`
	SubCategory string   `json:"sub_category,omitempty"`
	Tags        []string `json:"tags"`
	Confidence  float64  `json:"confidence"`
	Reasoning   string   `json:"reasoning"`
}

// DuplicateDetectionResult represents the result of AI-powered duplicate detection
type DuplicateDetectionResult struct {
	DuplicateGroups []DuplicateGroup `json:"duplicate_groups"`
	Confidence      float64          `json:"confidence"`
	Reasoning       string           `json:"reasoning"`
}

// DuplicateGroup represents a group of duplicate files
type DuplicateGroup struct {
	Files      []string `json:"files"`
	Similarity float64  `json:"similarity"`
	Type       string   `json:"type"` // "exact", "content_similar", "name_similar"
	Reasoning  string   `json:"reasoning"`
}
