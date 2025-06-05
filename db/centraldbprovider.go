package db

import (
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/ZanzyTHEbar/genkithandler"
	_ "github.com/tursodatabase/go-libsql"
)

// TODO: Fix logging so that it is optional and outputs to the default unix log directory, stdout|stderr, or the dfault config directory

// TODO: Implement generic database validation methods

// TODO: Implement default database settings

// CentralDBProvider tracks the locations of all workspaces.
type CentralDBProvider struct {
	db *sql.DB
}

// NewCentralDBProvider opens or initializes the central database at the binary location.
func NewCentralDBProvider() (*CentralDBProvider, error) {
	// Ensure the config directory exists
	if err := os.MkdirAll(genkithandler.DefaultConfigPath, 0755); err != nil {
		return nil, fmt.Errorf("could not create config directory: %v", err)
	}

	slog.Info("Central database path:", "path", genkithandler.DefaultCentralDBPath)

	db, err := ConnectToDB(genkithandler.DefaultCentralDBPath)
	if err != nil {
		return nil, err
	}

	provider := &CentralDBProvider{db: db}
	if err := provider.init(); err != nil {
		return nil, err
	}
	return provider, nil
}

// init sets up the central database tables.
func (c *CentralDBProvider) init() error {
	_, err := c.db.Exec(`CREATE TABLE IF NOT EXISTS workspaces (
		id TEXT PRIMARY KEY UNIQUE,
		root_path TEXT,
		config TEXT,
		time_stamp DATETIME DEFAULT CURRENT_TIMESTAMP
	)`)
	return err
}

// Close closes the central database connection.
func (c *CentralDBProvider) Close() error {
	return c.db.Close()
}

// Connect implements ICentralDBProvider.Connect
func (c *CentralDBProvider) Connect(dsn string) (*sql.DB, error) {
	var err error
	c.db, err = ConnectToDB(dsn)
	return c.db, err
}

// InitSchema implements ICentralDBProvider.InitSchema
func (c *CentralDBProvider) InitSchema() error {
	return c.init()
}

// Backup creates a backup of the central database.
// It returns the path to the backup file and any error that occurred during the process.
func (c *CentralDBProvider) Backup() (string, error) {
	if c.db == nil {
		return "", fmt.Errorf("cannot backup: database connection is nil")
	}

	backupDir := filepath.Join(genkithandler.DefaultConfigPath, "backups")
	if err := os.MkdirAll(backupDir, 0755); err != nil {
		return "", fmt.Errorf("could not create backup directory: %v", err)
	}

	// Generate unique backup filename with timestamp
	timestamp := time.Now().Format("20060102_150405")
	backupPath := filepath.Join(backupDir, fmt.Sprintf("central_backup_%s.db", timestamp))

	// Execute the backup using SQL VACUUM INTO command
	// This is specific to SQLite and creates a copy of the database
	_, err := c.db.Exec(fmt.Sprintf("VACUUM INTO '%s'", backupPath))
	if err != nil {
		return "", fmt.Errorf("backup failed: %v", err)
	}

	slog.Info("Database backup created successfully", "path", backupPath)
	return backupPath, nil
}
