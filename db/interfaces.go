package db

import (
	"database/sql"
)

// ICentralDBProvider is the interface for central database operations (using I prefix to avoid naming conflict)
type ICentralDBProvider interface {
	Connect(dsn string) (*sql.DB, error)
	Close() error
	InitSchema() error
	// TODO: Snapshot methods
	//InsertSnapshot(snapshot *Snapshot) (uuid.UUID, error)
	//GetSnapshot(id uuid.UUID) (*Snapshot, error)
	//GetLatestSnapshot() (*Snapshot, error)
	//GetAllSnapshots() ([]Snapshot, error)
	// Backup method
	Backup() (string, error)
}
