package providers

import (
	"context"
	"log/slog"

	"github.com/ZanzyTHEbar/genkithandler/pkg/config"
	"github.com/ZanzyTHEbar/genkithandler/pkg/domain"
	"github.com/firebase/genkit/go/ai"
)

// Manager handles multiple AI providers and provides fallback capabilities
type Manager struct {
	providers     map[ProviderType]AIProvider
	primaryType   ProviderType
	fallbackOrder []ProviderType
	initialized   bool
	logger        domain.Logger
	errorHandler  domain.ErrorHandler
}

// NewManager creates a new provider manager
func NewManager(logger domain.Logger, errorHandler domain.ErrorHandler) *Manager {
	return &Manager{
		providers:     make(map[ProviderType]AIProvider),
		fallbackOrder: []ProviderType{},
		logger:        logger,
		errorHandler:  errorHandler,
	}
}

// RegisterProvider registers an AI provider with the manager
func (m *Manager) RegisterProvider(providerType ProviderType, provider AIProvider) error {
	if provider == nil {
		return m.errorHandler.New("provider cannot be nil", map[string]interface{}{
			"provider_type": string(providerType),
		})
	}

	m.providers[providerType] = provider
	slog.Info("Registered AI provider", "type", string(providerType))

	return nil
}

// Initialize initializes all registered providers
func (m *Manager) Initialize(ctx context.Context, cfg config.Config) error {
	if m.initialized {
		return nil
	}

	// Initialize Google AI provider if configured
	if cfg.GoogleAI.APIKey != "" {
		googleaiProvider := NewGoogleAIProvider(m.logger, m.errorHandler)
		if err := m.RegisterProvider(ProviderTypeGoogleAI, googleaiProvider); err != nil {
			return err
		}

		// Create provider config map
		googleConfig := map[string]interface{}{
			"api_key":       cfg.GoogleAI.APIKey,
			"default_model": cfg.GoogleAI.DefaultModel,
		}

		if err := googleaiProvider.Initialize(ctx, googleConfig); err != nil {
			slog.Warn("Failed to initialize Google AI provider", "error", err)
		} else {
			m.fallbackOrder = append(m.fallbackOrder, ProviderTypeGoogleAI)
			if m.primaryType == "" {
				m.primaryType = ProviderTypeGoogleAI
			}
		}
	}

	if len(m.fallbackOrder) == 0 {
		return m.errorHandler.New("no AI providers are configured and available", map[string]interface{}{
			"configured_providers": len(m.providers),
		})
	}

	slog.Info("Provider manager initialized",
		"primary", string(m.primaryType),
		"fallback_order", m.fallbackOrder,
		"total_providers", len(m.providers))

	m.initialized = true
	return nil
}

// GenerateText generates text using the primary provider with fallback support
func (m *Manager) GenerateText(ctx context.Context, prompt string) (string, error) {
	if !m.initialized {
		return "", m.errorHandler.New("provider manager not initialized", map[string]interface{}{
			"method": "GenerateText",
		})
	}

	var lastErr error

	// Try primary provider first
	if provider, exists := m.providers[m.primaryType]; exists && provider.IsAvailable() {
		if result, err := provider.GenerateText(ctx, prompt); err == nil {
			return result, nil
		} else {
			lastErr = err
			slog.Warn("Primary provider failed", "provider", string(m.primaryType), "error", err)
		}
	}

	// Try fallback providers
	for _, providerType := range m.fallbackOrder {
		if providerType == m.primaryType {
			// Skip primary as we already tried it
			continue
		}

		if provider, exists := m.providers[providerType]; exists && provider.IsAvailable() {
			if result, err := provider.GenerateText(ctx, prompt); err == nil {
				slog.Info("Fallback provider succeeded", "provider", string(providerType))
				return result, nil
			} else {
				lastErr = err
				slog.Warn("Fallback provider failed", "provider", string(providerType), "error", err)
			}
		}
	}

	if lastErr != nil {
		return "", lastErr
	}

	return "", m.errorHandler.New("all AI providers failed to generate text", map[string]interface{}{
		"providers_tried": len(m.fallbackOrder),
		"method":          "GenerateText",
	})
}

// GenerateWithStructuredOutput generates structured output using the primary provider with fallback support
func (m *Manager) GenerateWithStructuredOutput(ctx context.Context, prompt string, outputType interface{}) (*ai.ModelResponse, error) {
	if !m.initialized {
		return nil, m.errorHandler.New("provider manager not initialized", map[string]interface{}{
			"method": "GenerateWithStructuredOutput",
		})
	}

	var lastErr error

	// Try primary provider first
	if provider, exists := m.providers[m.primaryType]; exists && provider.IsAvailable() {
		if result, err := provider.GenerateWithStructuredOutput(ctx, prompt, outputType); err == nil {
			return result, nil
		} else {
			lastErr = err
			slog.Warn("Primary provider failed for structured output", "provider", string(m.primaryType), "error", err)
		}
	}

	// Try fallback providers
	for _, providerType := range m.fallbackOrder {
		if providerType == m.primaryType {
			continue // Skip primary as we already tried it
		}

		if provider, exists := m.providers[providerType]; exists && provider.IsAvailable() {
			if result, err := provider.GenerateWithStructuredOutput(ctx, prompt, outputType); err == nil {
				slog.Info("Fallback provider succeeded for structured output", "provider", string(providerType))
				return result, nil
			} else {
				lastErr = err
				slog.Warn("Fallback provider failed for structured output", "provider", string(providerType), "error", err)
			}
		}
	}

	if lastErr != nil {
		return nil, lastErr
	}

	return nil, m.errorHandler.New("all AI providers failed to generate structured output", map[string]interface{}{
		"providers_tried": len(m.fallbackOrder),
		"method":          "GenerateWithStructuredOutput",
	})
}

// IsProviderAvailable checks if a specific provider is available
func (m *Manager) IsProviderAvailable(providerType ProviderType) bool {
	if provider, exists := m.providers[providerType]; exists {
		return provider.IsAvailable()
	}
	return false
}

// GetPrimaryProvider returns the primary provider type
func (m *Manager) GetPrimaryProvider() ProviderType {
	return m.primaryType
}

// SetPrimaryProvider sets the primary provider
func (m *Manager) SetPrimaryProvider(providerType ProviderType) error {
	if !m.IsProviderAvailable(providerType) {
		return m.errorHandler.New("provider is not available", map[string]interface{}{
			"provider_type": string(providerType),
			"method":        "IsProviderAvailable",
		})
	}
	m.primaryType = providerType
	return nil
}

// GetFallbackOrder returns the current fallback order
func (m *Manager) GetFallbackOrder() []ProviderType {
	return append([]ProviderType{}, m.fallbackOrder...) // Return a copy
}

// GetProvider returns a specific provider
func (m *Manager) GetProvider(providerType ProviderType) (AIProvider, bool) {
	provider, exists := m.providers[providerType]
	return provider, exists
}
