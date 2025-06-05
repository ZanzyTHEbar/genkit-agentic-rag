package types

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/viper"
)

// GenkitPlugin represents configuration for a Genkit plugin
type GenkitPlugin struct {
	APIKey         string `mapstructure:"apiKey"`
	DefaultModel   string `mapstructure:"defaultModel"`
	TimeoutSeconds int    `mapstructure:"timeoutSeconds"`
}

// GenkitPluginsConfig represents the configuration for all Genkit plugins
type GenkitPluginsConfig struct {
	GoogleAI GenkitPlugin `mapstructure:"googleAI"`
	OpenAI   GenkitPlugin `mapstructure:"openAI"`
}

// GenkitPromptsConfig represents the configuration for Genkit prompts
type GenkitPromptsConfig struct {
	Directory string `mapstructure:"directory"`
}

// GenkitConfig represents the main Genkit configuration
type GenkitConfig struct {
	Plugins GenkitPluginsConfig `mapstructure:"plugins"`
	Prompts GenkitPromptsConfig `mapstructure:"prompts"`
}

// DatabaseConfig represents database configuration
type DatabaseConfig struct {
	DSN  string `mapstructure:"dsn"`
	Type string `mapstructure:"type"`
}

// GenkitHandlerConfig represents handler-specific configuration
type GenkitHandlerConfig struct {
	FeatureFlags map[string]bool `mapstructure:"featureFlags"`
}

// AppConfig represents the main application configuration
type AppConfig struct {
	AppName          string              `mapstructure:"appName"`
	CacheDir         string              `mapstructure:"cacheDir"`
	ConfigPath       string              `mapstructure:"configPath"`
	GlobalConfigFile string              `mapstructure:"globalConfigFile"`
	GenkitHandler    GenkitHandlerConfig `mapstructure:"genkithandler"`
	Database         DatabaseConfig      `mapstructure:"database"`
	GenkitConfig     GenkitConfig        `mapstructure:"genkit"`
}

// LoadConfig reads configuration from file or environment variables.
func (a *AppConfig) LoadConfig(configPath string) (*AppConfig, error) {
	if configPath != "" {
		viper.SetConfigFile(configPath)
	} else {
		viper.AddConfigPath(".")
		viper.AddConfigPath("..")
		viper.AddConfigPath(filepath.Join("etc", a.AppName))
		viper.AddConfigPath(a.ConfigPath)
		viper.SetConfigName("config")
		viper.SetConfigType("yaml")
	}

	// Set default values
	viper.SetDefault("genkit.prompts.directory", "./prompts")
	viper.SetDefault("genkit.plugins.openai.timeoutSeconds", 60)

	viper.SetDefault(fmt.Sprintf("%s.cacheDir", a.AppName), a.CacheDir)
	viper.SetDefault(fmt.Sprintf("%s.database.dsn", a.AppName), a.Database.DSN)
	viper.SetDefault(fmt.Sprintf("%s.database.type", a.AppName), a.Database.Type)

	viper.AutomaticEnv()
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))

	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			// Config file not found; defaults will be used. This is not an error for the application to halt on.
			// It's good practice to log this situation if a logger is available here.
			// fmt.Printf("Warning: Config file not found at expected locations. Using default values. Searched: %s\n", viper.ConfigFileUsed())

			// TODO: Implement default configuration loading logic & config file generation.

		} else {
			return nil, fmt.Errorf("failed to read config file: %w", err)
		}
	}

	err := viper.Unmarshal(&a)
	if err != nil {
		return nil, fmt.Errorf("unable to decode into struct: %w", err)
	}

	return a, nil
}

// GetHomeDir returns the user's home directory or panics if unable to determine it.
func GetHomeDir() string {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic("Unable to get home directory: " + err.Error())
	}
	return homeDir
}
