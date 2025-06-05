package genkithandler

import (
	"path/filepath"

	"github.com/ZanzyTHEbar/genkithandler/types"
)

// NewAppConfig creates a new application configuration with the given app name.
func NewAppConfig(appName string) *types.AppConfig {
	name := getValidAppName(appName)
	configPath := filepath.Join(types.GetHomeDir(), ".config", name)

	return &types.AppConfig{
		AppName:          name,
		ConfigPath:       configPath,
		CacheDir:         filepath.Join(configPath, ".cache"),
		GlobalConfigFile: filepath.Join(configPath, "config.toml"),
		Database: types.DatabaseConfig{
			DSN:  "file:" + filepath.Join(configPath, "genkit.db"),
			Type: "sqlite3",
		},
	}
}

// getValidAppName ensures the app name is valid and non-empty.
func getValidAppName(name string) string {
	if name == "" {
		panic("App name cannot be empty")
	}

	// TODO: Additional validation could be added here if needed
	// For example: checking for invalid characters, length limits, etc.
	return name
}
