// Package genkithandler provides a simplified interface for integrating with Genkit.
package genkithandler

// TODO: Add method to register custom service actions and tools
// TODO: Add ability to register custom AI Response Parsing Functions
// TODO: Add ability to load and construct custom prompts dynamically using Genkit dotprompt support

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/ZanzyTHEbar/genkithandler/errors"
	"github.com/ZanzyTHEbar/genkithandler/providers"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// Service holds an initialized Genkit instance and provides methods
// for interacting with Genkit functionalities.
type Service struct {
	g               *genkit.Genkit
	cfg             GenkitHandlerConfig
	promptsDir      string
	loadedPrompts   map[string]Prompt
	providerManager *providers.Manager
}

// NewService initializes a new Genkit instance and returns a Service.
// It uses the globally loaded AppConfig.
func NewService(ctx context.Context, cdb *db.CentralDBProvider) (*Service, error) {

	appCfg := config.AppConfig

	// Load prompts
	prompts, err := LoadPrompts(appCfg.Genkit.Prompts.Directory)
	if err != nil {

		slog.Error("Failed to load prompts for Genkit handler", "directory", appCfg.Genkit.Prompts.Directory, "error", err)
		return nil, fmt.Errorf("failed to load prompts: %w", err)
	}

	// Initialize provider manager
	providerManager := providers.NewManager()

	// Prepare Genkit options based on configuration
	var genkitOpts []genkit.GenkitOption

	// Telemetry configuration has been deprioritized.
	// // Configure Telemetry
	// // This is illustrative; actual Genkit telemetry configuration might differ.
	// // Assuming OTLP (OpenTelemetry Protocol) exporter for traces and metrics.
	// if appCfg.Genkit.Telemetry.LoggingLevel != "" { // This line will cause a compile error as Telemetry field is removed
	// // Assuming Genkit has a way to set log level, perhaps via a telemetry option
	// // or a global setting. For this example, let's imagine an OTLP option.
	// slog.Info("Configuring Genkit Telemetry", "loggingLevel", appCfg.Genkit.Telemetry.LoggingLevel, "traceSampler", appCfg.Genkit.Telemetry.TraceSampler)
	// // TODO: Example: telemetryOpt, err := otlp.Init(ctx, otlp.Config{
	// // LoggingLevel: appCfg.Genkit.Telemetry.LoggingLevel,
	// // TraceSampler: appCfg.Genkit.Telemetry.TraceSampler,
	// // Endpoint: os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"), // Or from config
	// // })
	// // if err != nil {
	// // slog.Error("Failed to initialize OTLP telemetry for Genkit", "error", err)
	// // } else if telemetryOpt != nil {
	// // genkitOpts = append(genkitOpts, telemetryOpt)
	// // }
	// // TODO: For now, we'll just log, as specific GenkitOption for this is unknown.
	// }

	// Configure Plugins
	// Google AI Plugin
	if appCfg.Genkit.Plugins.GoogleAI.APIKey != "" {
		googleAIPlugin := &googlegenai.GoogleAI{
			APIKey: appCfg.Genkit.Plugins.GoogleAI.APIKey,
		}
		genkitOpts = append(genkitOpts, genkit.WithPlugins(googleAIPlugin))

		slog.Info("Google AI Plugin configured",
			"apiKeySet", appCfg.Genkit.Plugins.GoogleAI.APIKey != "",
			"model", appCfg.Genkit.Plugins.GoogleAI.DefaultModel)
	}

	// OpenAI Plugin
	if appCfg.Genkit.Plugins.OpenAI.APIKey != "" {
		// openAICfg := openai.Config{ // Uncomment when openai package is available
		// APIKey:         appCfg.Genkit.Plugins.OpenAI.APIKey,
		// DefaultModel:   appCfg.Genkit.Plugins.OpenAI.DefaultModel,
		// RequestTimeout: time.Duration(appCfg.Genkit.Plugins.OpenAI.TimeoutSeconds) * time.Second,
		// }
		// Similar to Google AI, the initialization pattern would depend on the Genkit SDK.
		// TODO: Example:
		// if err := openai.Init(ctx, openAICfg); err != nil {
		// slog.Error("Failed to initialize OpenAI plugin for Genkit", "error", err)
		// }
		// Or:
		// plugin, err := openai.NewPlugin(ctx, openAICfg)
		// if err != nil {
		// slog.Error("Failed to initialize OpenAI plugin", "error", err)
		// } else {
		// genkitOpts = append(genkitOpts, genkit.WithPlugin(plugin)) // Hypothetical option
		// }
		slog.Info("OpenAI Plugin configured (illustrative)", "apiKeySet", appCfg.Genkit.Plugins.OpenAI.APIKey != "", "model", appCfg.Genkit.Plugins.OpenAI.DefaultModel, "timeoutSeconds", appCfg.Genkit.Plugins.OpenAI.TimeoutSeconds)
	}

	// Initialize Genkit with the constructed options
	g, err := genkit.Init(ctx, genkitOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize Genkit: %w", err)
	}

	// Initialize AI provider manager
	if err := providerManager.Initialize(ctx, g, appCfg.Genkit); err != nil {
		slog.Warn("Failed to initialize AI provider manager", "error", err)
		// Continue without AI providers for now - allows the service to work without API keys
	}

	s := &Service{
		g:               g,
		cfg:             GenkitHandler,
		promptsDir:      appCfg.Genkit.Prompts.Directory,
		loadedPrompts:   prompts,
		providerManager: providerManager,
	}

	// Register tools before flows so flows can call tools
	if err := RegisterCoreTools(s.g, cdb); err != nil {
		slog.Error("Failed to register core tools", "error", err)
	}
	if err := RegisterCoreFlows(s.g); err != nil {
		slog.Error("Failed to register core flows", "error", err)
	}

	return s, nil
}

// GetProviderStatus returns the status of all AI providers
func (s *Service) GetProviderStatus() map[providers.ProviderType]bool {
	if s.providerManager == nil {
		return make(map[providers.ProviderType]bool)
	}
	return s.providerManager.GetProviderStatus()
}

// SetPrimaryProvider changes the primary AI provider
func (s *Service) SetPrimaryProvider(providerType providers.ProviderType) error {
	if s.providerManager == nil {
		return errors.New("AI provider manager not initialized")
	}
	return s.providerManager.SetPrimaryProvider(providerType)
}

// Genkit returns the underlying Genkit instance.
// This can be used for advanced scenarios where direct access to Genkit is needed.
func (s *Service) Genkit() *genkit.Genkit {
	return s.g
}

// GetPrompt retrieves a loaded prompt by its name.
// It returns the Prompt struct and a boolean indicating if the prompt was found.
func (s *Service) GetPrompt(name string) (Prompt, bool) {
	if s.loadedPrompts == nil {
		return Prompt{}, false
	}
	prompt, ok := s.loadedPrompts[name]
	return prompt, ok
}

// Close performs any cleanup required by the Service.
// For now, it's a placeholder.
func (s *Service) Close(ctx context.Context) error {
	// TODO: Add cleanup logic if necessary, e.g., for plugins or resources
	// that require explicit shutdown.
	return nil
}
