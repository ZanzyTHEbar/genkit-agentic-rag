package db

import (
	"time"

	"github.com/google/uuid"
)

type OperationType int

const (
	OperationTypeCreate OperationType = iota
	OperationTypeUpdate
	OperationTypeDelete
	OperationTypeMove
)

type OperationHistory struct {
	ID          uuid.UUID
	NodeID      uuid.UUID
	Operation   OperationType
	NewPath     string
	TimeStamp   time.Time
	PerformedBy string
}
