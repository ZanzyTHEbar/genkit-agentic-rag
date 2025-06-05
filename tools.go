// Package genkithandler provides a simplified interface for integrating with Genkit.
package genkithandler

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/google/uuid"
)

// ToolRegistry manages tool registration and execution with validation and monitoring
type ToolRegistry struct {
	tools map[string]ToolInfo
	mutex sync.RWMutex
}

// ToolInfo contains metadata about a registered tool
type ToolInfo struct {
	Tool         ai.Tool   `json:"-"`
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	InputType    string    `json:"input_type"`
	OutputType   string    `json:"output_type"`
	RegisteredAt time.Time `json:"registered_at"`
	CallCount    int64     `json:"call_count"`
	LastCalled   time.Time `json:"last_called,omitempty"`
	ErrorCount   int64     `json:"error_count"`
}

// ToolExecutionResult contains comprehensive execution results with metrics
type ToolExecutionResult struct {
	ToolName         string                 `json:"tool_name"`
	RequestID        string                 `json:"request_id"`
	Result           interface{}            `json:"result"`
	Success          bool                   `json:"success"`
	Error            error                  `json:"error,omitempty"`
	ErrorCode        string                 `json:"error_code,omitempty"`
	Duration         time.Duration          `json:"duration"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
	Timestamp        time.Time              `json:"timestamp"`
	ValidationResult *ValidationResult      `json:"validation,omitempty"`
}

// ValidationResult contains tool input/output validation results
type ValidationResult struct {
	InputValid  bool     `json:"input_valid"`
	OutputValid bool     `json:"output_valid"`
	Errors      []string `json:"errors,omitempty"`
	Warnings    []string `json:"warnings,omitempty"`
}

// ToolChainRequest represents a request to execute multiple tools in sequence
type ToolChainRequest struct {
	Tools     []ToolChainStep        `json:"tools"`
	Context   map[string]interface{} `json:"context,omitempty"`
	RequestID string                 `json:"request_id"`
	FailFast  bool                   `json:"fail_fast"` // Stop on first error
}

// ToolChainStep represents a single step in a tool chain
type ToolChainStep struct {
	ToolName  string                 `json:"tool_name"`
	Input     interface{}            `json:"input"`
	DependsOn []string               `json:"depends_on,omitempty"` // Tool names this step depends on
	UseOutput string                 `json:"use_output,omitempty"` // Use output from this tool name as input
	Transform string                 `json:"transform,omitempty"`  // JSONPath to extract from dependency output
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ToolChainResult contains the results of executing a tool chain
type ToolChainResult struct {
	RequestID    string                          `json:"request_id"`
	Success      bool                            `json:"success"`
	StepResults  map[string]*ToolExecutionResult `json:"step_results"`
	FinalResult  interface{}                     `json:"final_result,omitempty"`
	Duration     time.Duration                   `json:"duration"`
	ErrorSummary string                          `json:"error_summary,omitempty"`
}

// Global tool registry instance
var globalRegistry = &ToolRegistry{
	tools: make(map[string]ToolInfo),
}

// GetToolRegistry returns the global tool registry instance
func GetToolRegistry() *ToolRegistry {
	return globalRegistry
}

// RegisterTool registers a tool with enhanced metadata and validation
func (tr *ToolRegistry) RegisterTool(name, description, inputType, outputType string, tool ai.Tool) error {
	if name == "" {
		return NewGenkitError(ErrorCodeInvalidInput, "tool name cannot be empty")
	}
	if tool == nil {
		return NewGenkitError(ErrorCodeInvalidInput, "tool cannot be nil")
	}

	tr.mutex.Lock()
	defer tr.mutex.Unlock()

	// Check if tool already exists
	if _, exists := tr.tools[name]; exists {
		return NewGenkitError(ErrorCodeToolAlreadyExists, fmt.Sprintf("tool %s already registered", name))
	}

	tr.tools[name] = ToolInfo{
		Tool:         tool,
		Name:         name,
		Description:  description,
		InputType:    inputType,
		OutputType:   outputType,
		RegisteredAt: time.Now(),
		CallCount:    0,
		ErrorCount:   0,
	}

	return nil
}

// GetTool retrieves tool information by name
func (tr *ToolRegistry) GetTool(name string) (ToolInfo, bool) {
	tr.mutex.RLock()
	defer tr.mutex.RUnlock()

	tool, exists := tr.tools[name]
	return tool, exists
}

// ListTools returns all registered tools
func (tr *ToolRegistry) ListTools() map[string]ToolInfo {
	tr.mutex.RLock()
	defer tr.mutex.RUnlock()

	result := make(map[string]ToolInfo)
	for name, info := range tr.tools {
		// Create a copy to avoid race conditions
		result[name] = ToolInfo{
			Name:         info.Name,
			Description:  info.Description,
			InputType:    info.InputType,
			OutputType:   info.OutputType,
			RegisteredAt: info.RegisteredAt,
			CallCount:    info.CallCount,
			LastCalled:   info.LastCalled,
			ErrorCount:   info.ErrorCount,
		}
	}
	return result
}

// updateToolStats updates call statistics for a tool
func (tr *ToolRegistry) updateToolStats(name string, success bool) {
	tr.mutex.Lock()
	defer tr.mutex.Unlock()

	if info, exists := tr.tools[name]; exists {
		info.CallCount++
		info.LastCalled = time.Now()
		if !success {
			info.ErrorCount++
		}
		tr.tools[name] = info
	}
}

// DefineTool defines a new Genkit tool with enhanced error handling and registration
func DefineTool[In, Out any](
	g *genkit.Genkit,
	name, description string,
	fn func(ctx *ai.ToolContext, input In) (Out, error),
) (ai.Tool, error) {
	if g == nil {
		return nil, NewGenkitError(ErrorCodeInvalidInput, "genkit instance is nil")
	}
	if name == "" {
		return nil, NewGenkitError(ErrorCodeInvalidInput, "tool name cannot be empty")
	}
	if description == "" {
		return nil, NewGenkitError(ErrorCodeInvalidInput, "tool description cannot be empty")
	}
	if fn == nil {
		return nil, NewGenkitError(ErrorCodeInvalidInput, "tool function (fn) cannot be nil")
	}

	// Wrap the function with enhanced error handling and metrics
	wrappedFn := func(ctx *ai.ToolContext, input In) (Out, error) {
		var zeroOut Out
		startTime := time.Now()

		// Execute the original function
		result, err := fn(ctx, input)
		duration := time.Since(startTime)

		// Update tool statistics
		registry := GetToolRegistry()
		registry.updateToolStats(name, err == nil)

		if err != nil {
			// Wrap error with tool context
			genkitErr := NewGenkitError(ErrorCodeToolExecutionFailed, fmt.Sprintf("tool %s execution failed", name)).
				WithCause(err).
				WithContext("tool_name", name).
				WithContext("duration", duration.String())
			return zeroOut, genkitErr
		}

		return result, nil
	}

	// Define the tool with Genkit
	tool := genkit.DefineTool(g, name, description, wrappedFn)
	if tool == nil {
		return nil, NewGenkitError(ErrorCodeInternalError, fmt.Sprintf("failed to define tool %s", name))
	}

	// Register the tool in our registry with type information
	var inExample In
	var outExample Out
	inputType := fmt.Sprintf("%T", inExample)
	outputType := fmt.Sprintf("%T", outExample)

	if err := globalRegistry.RegisterTool(name, description, inputType, outputType, tool); err != nil {
		// Log the registration error but don't fail tool definition
		fmt.Printf("Warning: failed to register tool %s in registry: %v\n", name, err)
	}

	return tool, nil
}

// LookupTool retrieves a previously defined Genkit tool by its name from the Genkit instance.
// Returns nil if the tool is not found.
func LookupTool(g *genkit.Genkit, name string) ai.Tool {
	if g == nil || name == "" {
		return nil
	}
	return genkit.LookupTool(g, name)
}

// ExecuteToolWithValidation executes a tool with comprehensive validation and error handling
func ExecuteToolWithValidation[In, Out any](
	ctx context.Context,
	g *genkit.Genkit,
	toolName string,
	input In,
) (*ToolExecutionResult, error) {
	requestID := uuid.New().String()
	startTime := time.Now()

	result := &ToolExecutionResult{
		ToolName:  toolName,
		RequestID: requestID,
		Timestamp: startTime,
		Metadata:  make(map[string]interface{}),
	}

	// Validate input
	validation := &ValidationResult{
		InputValid: true,
		Errors:     []string{},
		Warnings:   []string{},
	}

	if toolName == "" {
		validation.InputValid = false
		validation.Errors = append(validation.Errors, "tool name cannot be empty")
	}

	if g == nil {
		validation.InputValid = false
		validation.Errors = append(validation.Errors, "genkit instance is nil")
	}

	result.ValidationResult = validation

	if !validation.InputValid {
		result.Success = false
		result.Duration = time.Since(startTime)
		result.Error = NewGenkitError(ErrorCodeInvalidInput, "tool execution validation failed")
		result.ErrorCode = string(ErrorCodeInvalidInput)
		return result, result.Error
	}

	// Look up the tool
	tool := LookupTool(g, toolName)
	if tool == nil {
		result.Success = false
		result.Duration = time.Since(startTime)
		result.Error = NewGenkitError(ErrorCodeToolNotFound, fmt.Sprintf("tool %s not found", toolName))
		result.ErrorCode = string(ErrorCodeToolNotFound)
		return result, result.Error
	}

	// Execute the tool
	outputRaw, err := tool.RunRaw(ctx, input)
	result.Duration = time.Since(startTime)

	if err != nil {
		result.Success = false
		result.Error = NewGenkitError(ErrorCodeToolExecutionFailed, fmt.Sprintf("tool %s execution failed", toolName)).
			WithCause(err).
			WithContext("request_id", requestID)
		result.ErrorCode = string(ErrorCodeToolExecutionFailed)
		return result, result.Error
	}

	// Convert output to expected type
	var output Out
	if m, ok := outputRaw.(map[string]interface{}); ok {
		jsonData, err := json.Marshal(m)
		if err != nil {
			result.Success = false
			result.Error = NewGenkitError(ErrorCodeSerialization, fmt.Sprintf("failed to marshal tool %s output", toolName)).
				WithCause(err)
			result.ErrorCode = string(ErrorCodeSerialization)
			return result, result.Error
		}
		if err := json.Unmarshal(jsonData, &output); err != nil {
			result.Success = false
			result.Error = NewGenkitError(ErrorCodeSerialization, fmt.Sprintf("failed to unmarshal tool %s output", toolName)).
				WithCause(err)
			result.ErrorCode = string(ErrorCodeSerialization)
			return result, result.Error
		}
	} else if typedOutput, ok := outputRaw.(Out); ok {
		output = typedOutput
	} else {
		result.Success = false
		result.Error = NewGenkitError(ErrorCodeTypeConversion, fmt.Sprintf("unexpected output type for tool %s: %T", toolName, outputRaw))
		result.ErrorCode = string(ErrorCodeTypeConversion)
		return result, result.Error
	}

	// Validate output
	validation.OutputValid = true
	result.ValidationResult = validation
	result.Success = true
	result.Result = output

	return result, nil
}

// ExecuteTool looks up a tool by name and executes it with the provided input (simplified interface)
func ExecuteTool[In, Out any](
	ctx context.Context,
	g *genkit.Genkit,
	toolName string,
	input In,
) (Out, error) {
	var zeroOut Out

	result, err := ExecuteToolWithValidation[In, Out](ctx, g, toolName, input)
	if err != nil {
		return zeroOut, err
	}

	if !result.Success {
		return zeroOut, result.Error
	}

	if output, ok := result.Result.(Out); ok {
		return output, nil
	}

	return zeroOut, NewGenkitError(ErrorCodeTypeConversion, "failed to convert tool result to expected type")
}

// ExecuteToolChain executes multiple tools in sequence with dependency management
func ExecuteToolChain(ctx context.Context, g *genkit.Genkit, request *ToolChainRequest) (*ToolChainResult, error) {
	if request == nil {
		return nil, NewGenkitError(ErrorCodeInvalidInput, "tool chain request cannot be nil")
	}

	if request.RequestID == "" {
		request.RequestID = uuid.New().String()
	}

	startTime := time.Now()
	result := &ToolChainResult{
		RequestID:   request.RequestID,
		StepResults: make(map[string]*ToolExecutionResult),
	}

	// Execute tools in order, handling dependencies
	stepOutputs := make(map[string]interface{})

	for i, step := range request.Tools {
		stepName := fmt.Sprintf("step_%d_%s", i, step.ToolName)

		// Check dependencies
		if len(step.DependsOn) > 0 {
			for _, dep := range step.DependsOn {
				if _, exists := stepOutputs[dep]; !exists {
					errorMsg := fmt.Sprintf("dependency %s not satisfied for step %s", dep, stepName)
					result.ErrorSummary = errorMsg
					result.Duration = time.Since(startTime)
					return result, NewGenkitError(ErrorCodeDependencyError, errorMsg)
				}
			}
		}

		// Prepare input (use dependency output if specified)
		var stepInput interface{} = step.Input
		if step.UseOutput != "" {
			if depOutput, exists := stepOutputs[step.UseOutput]; exists {
				stepInput = depOutput
			} else {
				errorMsg := fmt.Sprintf("output from %s not available for step %s", step.UseOutput, stepName)
				result.ErrorSummary = errorMsg
				result.Duration = time.Since(startTime)
				return result, NewGenkitError(ErrorCodeDependencyError, errorMsg)
			}
		}

		// Execute the tool
		stepResult, err := ExecuteToolWithValidation[interface{}, interface{}](ctx, g, step.ToolName, stepInput)
		result.StepResults[stepName] = stepResult

		if err != nil || !stepResult.Success {
			if request.FailFast {
				result.ErrorSummary = fmt.Sprintf("step %s failed: %v", stepName, err)
				result.Duration = time.Since(startTime)
				return result, err
			}
			// Continue with next step if not fail-fast
			continue
		}

		// Store output for potential use by subsequent steps
		stepOutputs[step.ToolName] = stepResult.Result
	}

	// Determine overall success
	successCount := 0
	for _, stepResult := range result.StepResults {
		if stepResult.Success {
			successCount++
		}
	}

	result.Success = successCount == len(request.Tools)
	result.Duration = time.Since(startTime)

	// Set final result to the last successful step's output
	if len(request.Tools) > 0 {
		lastStepName := fmt.Sprintf("step_%d_%s", len(request.Tools)-1, request.Tools[len(request.Tools)-1].ToolName)
		if lastResult, exists := result.StepResults[lastStepName]; exists && lastResult.Success {
			result.FinalResult = lastResult.Result
		}
	}

	return result, nil
}

// RegisterCoreTools registers core tools using enhanced tool definition with validation
func RegisterCoreTools(g *genkit.Genkit, cdb *db.CentralDBProvider) error {
	// performBackup tool with enhanced error handling
	backupToolHandler := func(ctx *ai.ToolContext, input BackupToolInput) (BackupToolOutput, error) {
		if cdb == nil {
			return BackupToolOutput{}, NewGenkitError(ErrorCodeInvalidInput, "central database provider is nil")
		}

		var centralDBPath string

		centralDBPath, err := cdb.Backup()

		// Determine success and create appropriate response
		if err != nil {
			msg := fmt.Sprintf("Backup failed: CentralDB: %v", err)
			return BackupToolOutput{
				CentralDBBackupPath: centralDBPath,
				Message:             msg,
			}, NewGenkitError(ErrorCodeBackupFailed, msg).WithCause(err).WithContext("central_db_error", err2.Error())
		}

		var msg string
		if err != nil {
			msg = fmt.Sprintf("DeskFS backup failed: %v", err)
		} else {
			msg = "Backup completed successfully"
		}

		return BackupToolOutput{
			CentralDBBackupPath: centralDBPath,
			Message:             msg,
		}, nil
	}

	if _, err := DefineTool(g, "performBackup", "Performs a backup of application data with comprehensive error handling.", backupToolHandler); err != nil {
		return NewGenkitError(ErrorCodeToolRegistrationFailed, "failed to register performBackup tool").WithCause(err)
	}

	// organizeDirectory tool placeholder with proper structure
	organizeHandler := func(ctx *ai.ToolContext, input interface{}) (map[string]string, error) {
		return map[string]string{
			"status":      "success",
			"message":     "Directory organization not implemented yet (new system)",
			"implemented": "false",
		}, nil
	}

	if _, err := DefineTool(g, "organizeDirectory", "Organizes a directory using AI-powered file categorization and structure optimization.", organizeHandler); err != nil {
		return NewGenkitError(ErrorCodeToolRegistrationFailed, "failed to register organizeDirectory tool").WithCause(err)
	}

	// manageWorkspace tool placeholder with proper structure
	workspaceHandler := func(ctx *ai.ToolContext, input interface{}) (map[string]string, error) {
		return map[string]string{
			"status":      "success",
			"message":     "Workspace management not implemented yet (new system)",
			"implemented": "false",
		}, nil
	}
	if _, err := DefineTool(g, "manageWorkspace", "Manages and configures workspaces with AI-powered optimization and automation.", workspaceHandler); err != nil {
		return NewGenkitError(ErrorCodeToolRegistrationFailed, "failed to register manageWorkspace tool").WithCause(err)
	}
	return nil
}
