package config

import (
	"fmt"
	"strings"

	errbuilder "github.com/ZanzyTHEbar/errbuilder-go"
	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
)

// Config represents the main configuration for the GenKit handler
type Config struct {
	// Server configuration
	Server ServerConfig `mapstructure:"server"`

	// Google AI configuration
	GoogleAI GoogleAIConfig `mapstructure:"google_ai"`

	// TursoDB configuration
	TursoDB TursoDBConfig `mapstructure:"turso_db"`

	// Vector store configuration
	VectorStore VectorStoreConfig `mapstructure:"vector_store"`

	// Agent configurations
	Agents map[string]AgentConfiguration `mapstructure:"agents"`

	// RAG configuration
	RAG RAGConfig `mapstructure:"rag"`

	// Session management configuration
	Session SessionConfig `mapstructure:"session"`

	// Logging configuration
	Logging LoggingConfig `mapstructure:"logging"`

	// Metrics configuration
	Metrics MetricsConfig `mapstructure:"metrics"`
}

// ServerConfig represents server-specific configuration
type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	IdleTimeout  int    `mapstructure:"idle_timeout"`
}

// GoogleAIConfig represents Google AI API configuration
type GoogleAIConfig struct {
	APIKey         string  `mapstructure:"api_key"`
	DefaultModel   string  `mapstructure:"default_model"`
	EmbeddingModel string  `mapstructure:"embedding_model"`
	Temperature    float32 `mapstructure:"temperature"`
	MaxTokens      int     `mapstructure:"max_tokens"`
	RequestTimeout int     `mapstructure:"request_timeout"`
	RetryAttempts  int     `mapstructure:"retry_attempts"`
	RetryDelay     int     `mapstructure:"retry_delay"`
}

// TursoDBConfig represents TursoDB configuration
type TursoDBConfig struct {
	DatabaseURL    string `mapstructure:"database_url"`
	AuthToken      string `mapstructure:"auth_token"`
	MaxConnections int    `mapstructure:"max_connections"`
	IdleTimeout    int    `mapstructure:"idle_timeout"`
	ConnTimeout    int    `mapstructure:"conn_timeout"`
}

// VectorStoreConfig represents vector store configuration
type VectorStoreConfig struct {
	DefaultStore     string                         `mapstructure:"default_store"`
	Stores           map[string]VectorStoreSettings `mapstructure:"stores"`
	EmbeddingDim     int                            `mapstructure:"embedding_dim"`
	SimilarityMetric string                         `mapstructure:"similarity_metric"`
}

// VectorStoreSettings represents settings for a specific vector store
type VectorStoreSettings struct {
	Type        string                 `mapstructure:"type"`
	Name        string                 `mapstructure:"name"`
	Description string                 `mapstructure:"description"`
	Config      map[string]interface{} `mapstructure:"config"`
}

// AgentConfiguration represents configuration for an agent
type AgentConfiguration struct {
	Name            string             `mapstructure:"name"`
	Description     string             `mapstructure:"description"`
	SystemPrompt    string             `mapstructure:"system_prompt"`
	ModelName       string             `mapstructure:"model_name"`
	Temperature     float32            `mapstructure:"temperature"`
	MaxTokens       int                `mapstructure:"max_tokens"`
	Tools           []string           `mapstructure:"tools"`
	VectorStoreName string             `mapstructure:"vector_store_name"`
	AgentType       string             `mapstructure:"agent_type"`
	Specializations []string           `mapstructure:"specializations"`
	ParentAgentID   string             `mapstructure:"parent_agent_id"`
	ChildAgentIDs   []string           `mapstructure:"child_agent_ids"`
	RetrievalConfig RAGRetrievalConfig `mapstructure:"retrieval_config"`
}

// RAGConfig represents RAG-specific configuration
type RAGConfig struct {
	DefaultRetrieval RAGRetrievalConfig `mapstructure:"default_retrieval"`
	ChunkSize        int                `mapstructure:"chunk_size"`
	ChunkOverlap     int                `mapstructure:"chunk_overlap"`
	MaxDocSize       int                `mapstructure:"max_doc_size"`
}

// RAGRetrievalConfig represents retrieval configuration for RAG
type RAGRetrievalConfig struct {
	TopK                int     `mapstructure:"top_k"`
	SimilarityThreshold float32 `mapstructure:"similarity_threshold"`
	MaxDocumentLength   int     `mapstructure:"max_document_length"`
	OverlapPercentage   float32 `mapstructure:"overlap_percentage"`
	RetrievalStrategy   string  `mapstructure:"retrieval_strategy"`
}

// SessionConfig represents session management configuration
type SessionConfig struct {
	Storage         string `mapstructure:"storage"`
	TTL             int    `mapstructure:"ttl"`
	CleanupInterval int    `mapstructure:"cleanup_interval"`
	MaxSessions     int    `mapstructure:"max_sessions"`
}

// LoggingConfig represents logging configuration
type LoggingConfig struct {
	Level      string `mapstructure:"level"`
	Format     string `mapstructure:"format"`
	Output     string `mapstructure:"output"`
	Structured bool   `mapstructure:"structured"`
}

// MetricsConfig represents metrics configuration
type MetricsConfig struct {
	Enabled   bool   `mapstructure:"enabled"`
	Provider  string `mapstructure:"provider"`
	Endpoint  string `mapstructure:"endpoint"`
	Namespace string `mapstructure:"namespace"`
}

// Manager implements the ConfigManager interface
type Manager struct {
	viper  *viper.Viper
	config *Config
}

// NewManager creates a new configuration manager
func NewManager() *Manager {
	v := viper.New()

	// Set default values
	setDefaults(v)

	return &Manager{
		viper:  v,
		config: &Config{},
	}
}

// setDefaults sets default configuration values
func setDefaults(v *viper.Viper) {
	// Server defaults
	v.SetDefault("server.host", "0.0.0.0")
	v.SetDefault("server.port", 8080)
	v.SetDefault("server.read_timeout", 30)
	v.SetDefault("server.write_timeout", 30)
	v.SetDefault("server.idle_timeout", 120)

	// Google AI defaults
	v.SetDefault("google_ai.default_model", "gemini-2.0-flash")
	v.SetDefault("google_ai.embedding_model", "text-embedding-004")
	v.SetDefault("google_ai.temperature", 0.7)
	v.SetDefault("google_ai.max_tokens", 4096)
	v.SetDefault("google_ai.request_timeout", 30)
	v.SetDefault("google_ai.retry_attempts", 3)
	v.SetDefault("google_ai.retry_delay", 1)

	// TursoDB defaults
	v.SetDefault("turso_db.max_connections", 10)
	v.SetDefault("turso_db.idle_timeout", 300)
	v.SetDefault("turso_db.conn_timeout", 10)

	// Vector store defaults
	v.SetDefault("vector_store.default_store", "default")
	v.SetDefault("vector_store.embedding_dim", 768)
	v.SetDefault("vector_store.similarity_metric", "cosine")

	// RAG defaults
	v.SetDefault("rag.default_retrieval.top_k", 5)
	v.SetDefault("rag.default_retrieval.similarity_threshold", 0.7)
	v.SetDefault("rag.default_retrieval.max_document_length", 4000)
	v.SetDefault("rag.default_retrieval.overlap_percentage", 0.1)
	v.SetDefault("rag.default_retrieval.retrieval_strategy", "simple")
	v.SetDefault("rag.chunk_size", 1000)
	v.SetDefault("rag.chunk_overlap", 200)
	v.SetDefault("rag.max_doc_size", 1000000)

	// Session defaults
	v.SetDefault("session.storage", "memory")
	v.SetDefault("session.ttl", 3600)
	v.SetDefault("session.cleanup_interval", 300)
	v.SetDefault("session.max_sessions", 10000)

	// Logging defaults
	v.SetDefault("logging.level", "info")
	v.SetDefault("logging.format", "json")
	v.SetDefault("logging.output", "stdout")
	v.SetDefault("logging.structured", true)

	// Metrics defaults
	v.SetDefault("metrics.enabled", false)
	v.SetDefault("metrics.provider", "prometheus")
	v.SetDefault("metrics.namespace", "genkithandler")
}

// Load loads configuration from files and environment variables
func (m *Manager) Load() error {
	// Set configuration file settings
	m.viper.SetConfigName("config")
	m.viper.SetConfigType("yaml")
	m.viper.AddConfigPath(".")
	m.viper.AddConfigPath("./config")
	m.viper.AddConfigPath("$HOME/.genkithandler")
	m.viper.AddConfigPath("/etc/genkithandler")

	// Enable environment variable support
	m.viper.AutomaticEnv()
	m.viper.SetEnvPrefix("GENKITHANDLER")
	m.viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	// Try to read configuration file
	if err := m.viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return errbuilder.GenericErr("failed to read config file", err)
		}
		// Config file not found is OK, we'll use defaults and env vars
	}

	// Unmarshal into config struct
	if err := m.viper.Unmarshal(m.config); err != nil {
		return errbuilder.GenericErr("failed to unmarshal config", err)
	}

	return nil
}

// Get returns the full configuration
func (m *Manager) Get() *Config {
	return m.config
}

// GetString returns a string configuration value
func (m *Manager) GetString(key string) string {
	return m.viper.GetString(key)
}

// GetInt returns an integer configuration value
func (m *Manager) GetInt(key string) int {
	return m.viper.GetInt(key)
}

// GetFloat32 returns a float32 configuration value
func (m *Manager) GetFloat32(key string) float32 {
	return float32(m.viper.GetFloat64(key))
}

// GetBool returns a boolean configuration value
func (m *Manager) GetBool(key string) bool {
	return m.viper.GetBool(key)
}

// GetStringMap returns a map configuration value
func (m *Manager) GetStringMap(key string) map[string]interface{} {
	return m.viper.GetStringMap(key)
}

// Validate validates the configuration
func (m *Manager) Validate() error {
	// Validate Google AI configuration
	if m.config.GoogleAI.APIKey == "" {
		return errbuilder.NewErrBuilder().WithMsg("Google AI API key is required")
	}

	// Validate TursoDB configuration
	if m.config.TursoDB.DatabaseURL == "" {
		return errbuilder.NewErrBuilder().WithMsg("TursoDB database URL is required")
	}

	if m.config.TursoDB.AuthToken == "" {
		return errbuilder.NewErrBuilder().WithMsg("TursoDB auth token is required")
	}

	// Validate vector store configuration
	if m.config.VectorStore.EmbeddingDim <= 0 {
		return errbuilder.NewErrBuilder().WithMsg("embedding dimension must be positive")
	}

	// Validate similarity metric
	validMetrics := []string{"cosine", "euclidean", "dot_product"}
	metric := m.config.VectorStore.SimilarityMetric
	valid := false
	for _, validMetric := range validMetrics {
		if metric == validMetric {
			valid = true
			break
		}
	}
	if !valid {
		return errbuilder.NewErrBuilder().WithMsg("invalid similarity metric")
	}

	// Validate agent configurations
	for _, agentConfig := range m.config.Agents {
		if agentConfig.Name == "" {
			return errbuilder.NewErrBuilder().WithMsg("agent name is required")
		}

		if agentConfig.SystemPrompt == "" {
			return errbuilder.NewErrBuilder().WithMsg("agent system prompt is required")
		}

		// Validate agent type
		validTypes := []string{"orchestrator", "specialist", "retrieval", "tool"}
		agentType := agentConfig.AgentType
		valid = false
		for _, validType := range validTypes {
			if agentType == validType {
				valid = true
				break
			}
		}
		if !valid {
			return errbuilder.NewErrBuilder().WithMsg("invalid agent type")
		}
	}

	return nil
}

// Set sets a configuration value
func (m *Manager) Set(key string, value interface{}) {
	m.viper.Set(key, value)
}

// Watch watches for configuration changes
func (m *Manager) Watch(callback func(key string, value interface{})) error {
	m.viper.WatchConfig()
	m.viper.OnConfigChange(func(e fsnotify.Event) {
		// Reload the configuration
		if err := m.viper.Unmarshal(m.config); err != nil {
			// Log error but don't fail
			fmt.Printf("Error reloading config: %v\n", err)
			return
		}
		callback("config", m.config)
	})
	return nil
}
