package providers

import (
	"context"
	"errors"
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
		return errors.New("provider cannot be nil")
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
		return errors.New("no AI providers are configured and available")
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
		return "", errors.New("provider manager not initialized")
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

	return "", errors.New("all AI providers failed to generate text")
}

// GenerateWithStructuredOutput generates structured output using the primary provider with fallback support
func (m *Manager) GenerateWithStructuredOutput(ctx context.Context, prompt string, outputType interface{}) (*ai.ModelResponse, error) {
	if !m.initialized {
		return nil, errors.New("provider manager not initialized")
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

	return nil, errors.New("all AI providers failed to generate structured output")
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
		return errors.New("provider is not available")
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
