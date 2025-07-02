package providers

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/ZanzyTHEbar/assert-lib"
	errbuilder "github.com/ZanzyTHEbar/errbuilder-go"
	"github.com/ZanzyTHEbar/genkithandler/pkg/domain"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// RetryConfig defines retry behavior for API calls
type RetryConfig struct {
	MaxRetries int
	BaseDelay  time.Duration
	MaxDelay   time.Duration
}

type GoogleAIProviderConfig struct {
	APIKey       string `json:"api_key" mapstructure:"api_key"`
	DefaultModel string `json:"default_model" mapstructure:"default_model"`
}

// GoogleAIProvider represents the Google AI provider configuration using Genkit's built-in plugin
// Store genkit instance
type GoogleAIProvider struct {
	initialized  bool
	retryConfig  RetryConfig
	config       GoogleAIProviderConfig
	logger       domain.Logger
	errorHandler domain.ErrorHandler
	genkit       *genkit.Genkit
}

// NewGoogleAIProvider creates a new Google AI provider instance
func NewGoogleAIProvider(logger domain.Logger, errorHandler domain.ErrorHandler) *GoogleAIProvider {
	// Default retry configuration
	retryConfig := RetryConfig{
		MaxRetries: 3,
		BaseDelay:  1 * time.Second,
		MaxDelay:   30 * time.Second,
	}

	return &GoogleAIProvider{
		retryConfig:  retryConfig,
		logger:       logger,
		errorHandler: errorHandler,
		initialized:  false,
	}
}

// Initialize initializes the Google AI provider using GenKit's plugin system
func (p *GoogleAIProvider) Initialize(ctx context.Context, config map[string]interface{}) error {
	// Parse configuration
	var googleConfig GoogleAIProviderConfig
	if err := p.parseConfig(config, &googleConfig); err != nil {
		return p.errorHandler.Wrap(err, "failed to parse configuration", map[string]interface{}{
			"config": config,
		})
	}

	// Validate required fields
	if googleConfig.APIKey == "" {
		return p.errorHandler.New("Google AI API key is required", map[string]interface{}{
			"config": config,
		})
	}

	if googleConfig.DefaultModel == "" {
		googleConfig.DefaultModel = "gemini-1.5-flash" // Default model
	}

	p.config = googleConfig

	// Initialize GenKit with GoogleAI plugin - proper pattern according to documentation
	genkitInstance, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{
		APIKey: googleConfig.APIKey,
	}))
	if err != nil {
		return p.errorHandler.Wrap(err, "failed to initialize Genkit with GoogleAI plugin", map[string]interface{}{
			"api_key_length": len(googleConfig.APIKey),
		})
	}

	p.genkit = genkitInstance
	p.initialized = true

	p.logger.Info("Google AI provider initialized successfully", map[string]interface{}{
		"default_model": googleConfig.DefaultModel,
	})

	return nil
}

// GetModel returns the configured model for Google AI
func (p *GoogleAIProvider) GetModel() string {
	if p.config.DefaultModel == "" {
		return "gemini-1.5-flash"
	}
	return p.config.DefaultModel
}

// GenerateText generates text using the Google AI provider
func (p *GoogleAIProvider) GenerateText(ctx context.Context, prompt string) (string, error) {
	if !p.initialized {
		return "", p.errorHandler.New("provider not initialized", map[string]interface{}{
			"provider": "googleai",
			"method":   "GenerateText",
		})
	}

	// Input validation using assert-lib
	assert.NotEmpty(ctx, prompt, "prompt cannot be empty", assert.WithPanicOnFailure())

	// Use the built-in Genkit generate function with the Google AI model
	response, err := p.withRetry(ctx, func() (string, error) {
		// Get the model reference from the plugin
		model := googlegenai.GoogleAIModel(p.genkit, p.GetModel())

		// Use the correct GenKit API - GenerateText is the correct function
		result, err := genkit.GenerateText(ctx, p.genkit,
			ai.WithModel(model),
			ai.WithPrompt(prompt),
		)

		return result, err
	})

	return response, err
}

// GenerateWithStructuredOutput generates text with structured output using the Google AI provider
func (p *GoogleAIProvider) GenerateWithStructuredOutput(ctx context.Context, prompt string, outputType interface{}) (*ai.ModelResponse, error) {
	if !p.initialized {
		return nil, p.errorHandler.New("provider not initialized", map[string]interface{}{
			"provider":    "googleai",
			"method":      "GenerateWithStructuredOutput",
			"output_type": fmt.Sprintf("%T", outputType),
		})
	}

	// Use the built-in Genkit generate function with structured output
	return p.withRetryStructured(ctx, func() (*ai.ModelResponse, error) {
		// Get the model reference from the plugin
		model := googlegenai.GoogleAIModel(p.genkit, p.GetModel())

		// Use the correct GenKit API - Generate with OutputType
		result, err := genkit.Generate(ctx, p.genkit,
			ai.WithModel(model),
			ai.WithPrompt(prompt),
			ai.WithOutputType(outputType),
		)

		return result, err
	})
}

// withRetry implements retry logic for text generation
func (p *GoogleAIProvider) withRetry(ctx context.Context, fn func() (string, error)) (string, error) {
	var lastErr error

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate exponential backoff delay
			delay := min(p.retryConfig.BaseDelay*time.Duration(1<<(attempt-1)), p.retryConfig.MaxDelay)

			slog.Debug("Retrying Google AI request",
				"attempt", attempt,
				"delay", delay.String(),
				"last_error", lastErr.Error())

			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case <-time.After(delay):
				// Continue with retry
			}
		}

		result, err := fn()
		if err == nil {
			return result, nil
		}

		lastErr = err

		// Check if error is retryable
		if !p.isRetryable(err) {
			break
		}
	}

	return "", fmt.Errorf("google AI request failed after %d attempts: %w", p.retryConfig.MaxRetries+1, lastErr)
}

// withRetryStructured implements retry logic for structured generation
func (p *GoogleAIProvider) withRetryStructured(ctx context.Context, fn func() (*ai.ModelResponse, error)) (*ai.ModelResponse, error) {
	var lastErr error

	for attempt := 0; attempt <= p.retryConfig.MaxRetries; attempt++ {
		if attempt > 0 {
			// Calculate exponential backoff delay
			delay := min(p.retryConfig.BaseDelay*time.Duration(1<<(attempt-1)), p.retryConfig.MaxDelay)

			slog.Debug("Retrying Google AI structured request",
				"attempt", attempt,
				"delay", delay.String(),
				"last_error", lastErr.Error())

			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(delay):
				// Continue with retry
			}
		}

		result, err := fn()
		if err == nil {
			return result, nil
		}

		lastErr = err

		// Check if error is retryable
		if !p.isRetryable(err) {
			break
		}
	}

	return nil, fmt.Errorf("google AI structured request failed after %d attempts: %w", p.retryConfig.MaxRetries+1, lastErr)
}

// isRetryable determines if an error should trigger a retry
func (p *GoogleAIProvider) isRetryable(err error) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())

	// Retryable conditions for Google AI API
	retryablePatterns := []string{
		"rate limit",
		"too many requests",
		"quota exceeded",
		"service unavailable",
		"internal error",
		"timeout",
		"connection reset",
		"temporary failure",
		"server error",
		"resource exhausted",
	}

	for _, pattern := range retryablePatterns {
		if strings.Contains(errStr, pattern) {
			return true
		}
	}

	return false
}

// IsAvailable checks if the Google AI provider is available and configured
func (p *GoogleAIProvider) IsAvailable() bool {
	return p.config.APIKey != "" && p.initialized
}

// SupportsStructuredOutput indicates whether this provider supports structured output
func (p *GoogleAIProvider) SupportsStructuredOutput() bool {
	return true // Google AI/Gemini supports structured output
}

// GetMaxTokens returns the maximum token limit for the configured model
func (p *GoogleAIProvider) GetMaxTokens() int {
	// Return conservative limits for different Gemini models
	model := p.GetModel()
	switch model {
	case "gemini-1.5-pro", "gemini-1.5-pro-latest":
		return 2097152
	case "gemini-1.5-flash", "gemini-1.5-flash-latest":
		return 1048576
	case "gemini-2.0-flash":
		return 1048576
	default:
		return 32768
	}
}

// GenerateStream generates a streaming response using GenKit's native streaming
func (p *GoogleAIProvider) GenerateStream(ctx context.Context, g *genkit.Genkit, prompt string) (<-chan StreamChunk, error) {
	if !p.initialized {
		resultChan := make(chan StreamChunk, 1)
		close(resultChan)
		return resultChan, p.errorHandler.New("provider not initialized", map[string]interface{}{
			"provider": "googleai",
			"method":   "GenerateStream",
		})
	}

	resultChan := make(chan StreamChunk, 10)

	// Launch goroutine to handle streaming using GenKit's native streaming
	go func() {
		defer close(resultChan)

		// Get the model reference from the plugin
		model := googlegenai.GoogleAIModel(p.genkit, p.GetModel())

		// Use GenKit's native streaming capabilities
		_, err := genkit.Generate(ctx, p.genkit,
			ai.WithModel(model),
			ai.WithPrompt(prompt),
			ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				// Convert GenKit chunk to our StreamChunk format
				streamChunk := StreamChunk{
					Content: chunk.Text(),
					Done:    false,
					Delta: map[string]interface{}{
						"index": chunk.Index,
					},
					Metadata: map[string]interface{}{
						"provider": "googleai",
						"model":    p.GetModel(),
					},
				}

				select {
				case resultChan <- streamChunk:
					return nil
				case <-ctx.Done():
					return ctx.Err()
				}
			}),
		)

		if err != nil {
			resultChan <- StreamChunk{
				Content: "",
				Done:    true,
				Error:   err,
				Metadata: map[string]interface{}{
					"provider": "googleai",
					"model":    p.GetModel(),
				},
			}
			return
		}

		// Send final completion chunk
		resultChan <- StreamChunk{
			Content: "",
			Done:    true,
			Metadata: map[string]interface{}{
				"provider":  "googleai",
				"model":     p.GetModel(),
				"completed": true,
			},
		}
	}()

	return resultChan, nil
}

// CallTool executes a tool through the AI model using GenKit's tool calling system
func (p *GoogleAIProvider) CallTool(ctx context.Context, g *genkit.Genkit, toolName string, params map[string]interface{}) (*ToolCallResult, error) {
	if !p.initialized {
		return nil, p.errorHandler.New("provider not initialized", map[string]interface{}{
			"provider":  "googleai",
			"method":    "CallTool",
			"tool_name": toolName,
		})
	}

	startTime := time.Now()

	// Validate inputs
	if toolName == "" {
		return &ToolCallResult{
			Result:   nil,
			Success:  false,
			Duration: time.Since(startTime),
			Error:    fmt.Errorf("tool name cannot be empty"),
		}, fmt.Errorf("tool name cannot be empty")
	}

	// Look up the tool from GenKit's registry
	tool := genkit.LookupTool(p.genkit, toolName)
	if tool == nil {
		return &ToolCallResult{
			Result:   nil,
			Success:  false,
			Duration: time.Since(startTime),
			Error:    fmt.Errorf("tool %q not found", toolName),
			Metadata: map[string]interface{}{
				"provider":  "googleai",
				"tool_name": toolName,
			},
		}, fmt.Errorf("tool %q not found", toolName)
	}

	// Execute the tool using GenKit's tool execution
	result, err := tool.RunRaw(ctx, params)
	duration := time.Since(startTime)

	if err != nil {
		return &ToolCallResult{
			Result:   nil,
			Success:  false,
			Duration: duration,
			Error:    err,
			Metadata: map[string]interface{}{
				"provider":   "googleai",
				"tool_name":  toolName,
				"error_type": "tool_execution_failed",
			},
		}, err
	}

	// Create successful result
	toolResult := &ToolCallResult{
		Result:   result,
		Success:  true,
		Duration: duration,
		Metadata: map[string]interface{}{
			"provider":  "googleai",
			"tool_name": toolName,
		},
	}

	return toolResult, nil
}

// parseConfig parses the configuration map into GoogleAIProviderConfig struct
func (p *GoogleAIProvider) parseConfig(config map[string]interface{}, googleConfig *GoogleAIProviderConfig) error {
	// Validate input parameters
	if config == nil {
		return errbuilder.New().
			WithMsg("config cannot be nil").
			WithCode(errbuilder.CodeInvalidArgument)
	}

	if googleConfig == nil {
		return errbuilder.New().
			WithMsg("googleConfig cannot be nil").
			WithCode(errbuilder.CodeInvalidArgument)
	}

	// Parse API key with type validation
	if apiKeyVal, ok := config["api_key"]; ok {
		if apiKey, ok := apiKeyVal.(string); ok {
			googleConfig.APIKey = apiKey
		} else {
			return errbuilder.New().
				WithMsg("Invalid api_key type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("string"),
					"actual_type":   fmt.Errorf("%T", apiKeyVal),
					"value":         fmt.Errorf("%v", apiKeyVal),
				}))
		}
	}

	// Parse default model with type validation
	if defaultModelVal, ok := config["default_model"]; ok {
		if defaultModel, ok := defaultModelVal.(string); ok {
			googleConfig.DefaultModel = defaultModel
		} else {
			return errbuilder.New().
				WithMsg("Invalid default_model type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("string"),
					"actual_type":   fmt.Errorf("%T", defaultModelVal),
					"value":         fmt.Errorf("%v", defaultModelVal),
				}))
		}
	}

	// Parse embedding model with type validation (if exists in config struct)
	if embeddingModelVal, ok := config["embedding_model"]; ok {
		if embeddingModel, ok := embeddingModelVal.(string); ok {
			// FIXME: GoogleAIProviderConfig doesn't have EmbeddingModel field yet
			// This would require adding it to the struct definition
			_ = embeddingModel // Silently ignore for now
		} else {
			return errbuilder.New().
				WithMsg("Invalid embedding_model type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("string"),
					"actual_type":   fmt.Errorf("%T", embeddingModelVal),
					"value":         fmt.Errorf("%v", embeddingModelVal),
				}))
		}
	}

	// Parse temperature with type validation (if exists in config struct)
	if temperatureVal, ok := config["temperature"]; ok {
		switch v := temperatureVal.(type) {
		case float32:
			// FIXME: GoogleAIProviderConfig doesn't have Temperature field yet
			_ = v // Silently ignore for now
		case float64:
			_ = float32(v) // Silently ignore for now
		case int:
			_ = float32(v) // Silently ignore for now
		default:
			return errbuilder.New().
				WithMsg("Invalid temperature type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("float32/float64/int"),
					"actual_type":   fmt.Errorf("%T", temperatureVal),
					"value":         fmt.Errorf("%v", temperatureVal),
				}))
		}
	}

	// Parse max tokens with type validation (if exists in config struct)
	if maxTokensVal, ok := config["max_tokens"]; ok {
		switch v := maxTokensVal.(type) {
		case int:
			// FIXME: GoogleAIProviderConfig doesn't have MaxTokens field yet
			_ = v // Silently ignore for now
		case float64:
			_ = int(v) // Silently ignore for now
		case float32:
			_ = int(v) // Silently ignore for now
		default:
			return errbuilder.New().
				WithMsg("Invalid max_tokens type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("int/float64/float32"),
					"actual_type":   fmt.Errorf("%T", maxTokensVal),
					"value":         fmt.Errorf("%v", maxTokensVal),
				}))
		}
	}

	// Parse request timeout with type validation (if exists in config struct)
	if requestTimeoutVal, ok := config["request_timeout"]; ok {
		switch v := requestTimeoutVal.(type) {
		case int:
			// FIXME: GoogleAIProviderConfig doesn't have RequestTimeout field yet
			_ = v // Silently ignore for now
		case float64:
			_ = int(v) // Silently ignore for now
		case float32:
			_ = int(v) // Silently ignore for now
		default:
			return errbuilder.New().
				WithMsg("Invalid request_timeout type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("int/float64/float32"),
					"actual_type":   fmt.Errorf("%T", requestTimeoutVal),
					"value":         fmt.Errorf("%v", requestTimeoutVal),
				}))
		}
	}

	// Parse retry attempts with type validation (if exists in config struct)
	if retryAttemptsVal, ok := config["retry_attempts"]; ok {
		switch v := retryAttemptsVal.(type) {
		case int:
			// FIXME: GoogleAIProviderConfig doesn't have RetryAttempts field yet
			_ = v // Silently ignore for now
		case float64:
			_ = int(v) // Silently ignore for now
		case float32:
			_ = int(v) // Silently ignore for now
		default:
			return errbuilder.New().
				WithMsg("Invalid retry_attempts type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("int/float64/float32"),
					"actual_type":   fmt.Errorf("%T", retryAttemptsVal),
					"value":         fmt.Errorf("%v", retryAttemptsVal),
				}))
		}
	}

	// Parse retry delay with type validation (if exists in config struct)
	if retryDelayVal, ok := config["retry_delay"]; ok {
		switch v := retryDelayVal.(type) {
		case int:
			// FIXME: GoogleAIProviderConfig doesn't have RetryDelay field yet
			_ = v // Silently ignore for now
		case float64:
			_ = int(v) // Silently ignore for now
		case float32:
			_ = int(v) // Silently ignore for now
		default:
			return errbuilder.New().
				WithMsg("Invalid retry_delay type").
				WithCode(errbuilder.CodeInvalidArgument).
				WithDetails(errbuilder.NewErrDetails(errbuilder.ErrorMap{
					"expected_type": fmt.Errorf("int/float64/float32"),
					"actual_type":   fmt.Errorf("%T", retryDelayVal),
					"value":         fmt.Errorf("%v", retryDelayVal),
				}))
		}
	}

	return nil
}
