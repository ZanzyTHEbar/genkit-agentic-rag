package providers

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/ZanzyTHEbar/genkithandler/pkg/domain"
	"github.com/google/uuid"
	"github.com/tursodatabase/libsql-client-go/libsql"
)

// TursoVectorStore implements the VectorStore interface using TursoDB
type TursoVectorStore struct {
	db             *sql.DB
	connector      driver.Connector
	tableName      string
	embedDimension int
	logger         domain.Logger
	errorHandler   domain.ErrorHandler
}

// TursoConfig represents configuration for TursoDB vector store
type TursoConfig struct {
	URL            string `json:"url" mapstructure:"url"`
	AuthToken      string `json:"auth_token" mapstructure:"auth_token"`
	TableName      string `json:"table_name" mapstructure:"table_name"`
	EmbedDimension int    `json:"embed_dimension" mapstructure:"embed_dimension"`
	CreateTable    bool   `json:"create_table" mapstructure:"create_table"`
}

// NewTursoVectorStore creates a new TursoDB vector store instance
func NewTursoVectorStore(logger domain.Logger, errorHandler domain.ErrorHandler) *TursoVectorStore {
	return &TursoVectorStore{
		logger:       logger,
		errorHandler: errorHandler,
		tableName:    "documents",
	}
}

// Initialize initializes the TursoDB vector store with the given configuration
func (t *TursoVectorStore) Initialize(ctx context.Context, config map[string]interface{}) error {
	// Parse configuration
	var tursoConfig TursoConfig
	if err := t.parseConfig(config, &tursoConfig); err != nil {
		return t.errorHandler.Wrap(err, "failed to parse configuration", map[string]interface{}{
			"config": config,
		})
	}

	// Validate required fields
	if tursoConfig.URL == "" {
		return t.errorHandler.New("TursoDB URL is required", map[string]interface{}{
			"config": config,
		})
	}

	if tursoConfig.AuthToken == "" {
		return t.errorHandler.New("TursoDB auth token is required", map[string]interface{}{
			"config": config,
		})
	}

	if tursoConfig.EmbedDimension <= 0 {
		tursoConfig.EmbedDimension = 384 // Default dimension for most embedding models
	}

	if tursoConfig.TableName != "" {
		t.tableName = tursoConfig.TableName
	}

	t.embedDimension = tursoConfig.EmbedDimension

	// Create connector using the correct libSQL pattern
	connector, err := libsql.NewConnector(tursoConfig.URL, libsql.WithAuthToken(tursoConfig.AuthToken))
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to create TursoDB connector", map[string]interface{}{
			"url":        tursoConfig.URL,
			"table_name": tursoConfig.TableName,
		})
	}

	t.connector = connector

	// Open database connection
	db := sql.OpenDB(connector)
	if err := db.PingContext(ctx); err != nil {
		return t.errorHandler.Wrap(err, "failed to connect to TursoDB", map[string]interface{}{
			"url":        tursoConfig.URL,
			"table_name": tursoConfig.TableName,
		})
	}

	t.db = db

	// Create table if requested
	if tursoConfig.CreateTable {
		if err := t.createVectorTable(ctx); err != nil {
			return t.errorHandler.Wrap(err, "failed to create vector table", map[string]interface{}{
				"table_name": t.tableName,
			})
		}
	}

	return nil
}

// createVectorTable creates the vector table with proper schema
func (t *TursoVectorStore) createVectorTable(ctx context.Context) error {
	// Create table with F32_BLOB for vector storage (library-compatible)
	createTableSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id TEXT PRIMARY KEY,
			content TEXT NOT NULL,
			metadata JSON,
			embedding F32_BLOB(%d),
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`, t.tableName, t.embedDimension)

	if _, err := t.db.ExecContext(ctx, createTableSQL); err != nil {
		return err
	}

	// Create vector index for efficient similarity search
	createIndexSQL := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS %s_vec_idx 
		ON %s(libsql_vector_idx(embedding, 'metric=cosine'))`,
		t.tableName, t.tableName)

	if _, err := t.db.ExecContext(ctx, createIndexSQL); err != nil {
		return err
	}

	return nil
}

// Store stores documents with their embeddings
func (t *TursoVectorStore) Store(ctx context.Context, documents []domain.Document) error {
	if len(documents) == 0 {
		return nil
	}

	// Start transaction
	tx, err := t.db.BeginTx(ctx, nil)
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to begin transaction", map[string]interface{}{
			"documents_count": len(documents),
		})
	}
	defer tx.Rollback()

	// Prepare insert statement using vector32() for proper vector storage
	query := fmt.Sprintf(`
		INSERT OR REPLACE INTO %s (
			id, content, metadata, embedding, created_at, updated_at
		) VALUES (?, ?, ?, vector32(?), ?, ?)
	`, t.tableName)

	stmt, err := tx.PrepareContext(ctx, query)
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to prepare statement", map[string]interface{}{
			"query":           query,
			"documents_count": len(documents),
		})
	}
	defer stmt.Close()

	// Insert documents
	for _, doc := range documents {
		if doc.ID == "" {
			doc.ID = uuid.New().String()
		}

		if doc.CreatedAt.IsZero() {
			doc.CreatedAt = time.Now()
		}
		doc.UpdatedAt = time.Now()

		// Serialize metadata
		metadataJSON, err := json.Marshal(doc.Metadata)
		if err != nil {
			return t.errorHandler.Wrap(err, "failed to serialize metadata", map[string]interface{}{
				"document_id": doc.ID,
				"metadata":    doc.Metadata,
			})
		}

		// Convert embedding to JSON string for vector32() function
		embeddingJSON, err := json.Marshal(doc.Embedding)
		if err != nil {
			return t.errorHandler.Wrap(err, "failed to serialize embedding", map[string]interface{}{
				"document_id": doc.ID,
				"embedding":   len(doc.Embedding),
			})
		}

		_, err = stmt.ExecContext(ctx,
			doc.ID,
			doc.Content,
			string(metadataJSON),
			string(embeddingJSON), // This will be passed to vector32() function
			doc.CreatedAt,
			doc.UpdatedAt,
		)
		if err != nil {
			return t.errorHandler.Wrap(err, "failed to insert document", map[string]interface{}{
				"document_id": doc.ID,
			})
		}
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return t.errorHandler.Wrap(err, "failed to commit transaction", map[string]interface{}{
			"documents_count": len(documents),
		})
	}

	t.logger.Info("Successfully stored documents", map[string]interface{}{
		"documents_count": len(documents),
		"table_name":      t.tableName,
	})

	return nil
}

// Search searches for similar documents using Turso's native vector functions
func (t *TursoVectorStore) Search(ctx context.Context, query domain.Query) (*domain.QueryResult, error) {
	if len(query.Embedding) == 0 {
		return nil, t.errorHandler.New("embedding is required for search", map[string]interface{}{
			"query": query.Text,
		})
	}

	// Set default limit
	limit := query.Limit
	if limit <= 0 {
		limit = 10
	}

	// Convert embedding to JSON for vector32() function
	embeddingJSON, err := json.Marshal(query.Embedding)
	if err != nil {
		return nil, t.errorHandler.Wrap(err, "failed to serialize query embedding", map[string]interface{}{
			"embedding_length": len(query.Embedding),
		})
	}

	// Try index-based search first for better performance
	var documents []domain.Document
	var scores []float32

	// Index-based search using vector_top_k
	indexQuerySQL := fmt.Sprintf(`
		SELECT 
			d.id, d.content, d.metadata, d.embedding, d.created_at, d.updated_at,
			d.source, d.chunk_index, d.total_chunks,
			vector_distance_cos(d.embedding, vector32(?)) as distance
		FROM vector_top_k('%s_vec_idx', vector32(?), ?) vtk
		JOIN %s d ON d.rowid = vtk.id
		ORDER BY distance ASC
	`, t.tableName, t.tableName)

	indexRows, indexErr := t.db.QueryContext(ctx, indexQuerySQL,
		string(embeddingJSON), string(embeddingJSON), limit)

	if indexErr == nil {
		defer indexRows.Close()
		documents, scores, err = t.parseSearchResults(indexRows, query.Threshold, true)
		if err == nil && len(documents) > 0 {
			result := &domain.QueryResult{
				Documents: documents,
				Scores:    scores,
				Total:     len(documents),
				Query:     query,
			}

			t.logger.Debug("Index-based search completed successfully", map[string]interface{}{
				"query":                query.Text,
				"results_count":        len(documents),
				"similarity_threshold": query.Threshold,
			})

			return result, nil
		}
		indexRows.Close()
	}

	// Fallback to direct similarity search if index search fails
	t.logger.Debug("Falling back to direct similarity search", map[string]interface{}{
		"index_error": indexErr,
	})

	// Build WHERE clause for filters
	whereClause, args := t.buildWhereClause(query.Filters)

	// Direct search using vector_distance_cos
	directQuerySQL := fmt.Sprintf(`
		SELECT 
			id, content, metadata, embedding, created_at, updated_at,
			source, chunk_index, total_chunks,
			vector_distance_cos(embedding, vector32(?)) as distance
		FROM %s
		%s
		ORDER BY distance ASC
		LIMIT ?
	`, t.tableName, whereClause)

	// Prepare arguments: embedding first, then filter args, then limit
	queryArgs := []interface{}{string(embeddingJSON)}
	queryArgs = append(queryArgs, args...)
	queryArgs = append(queryArgs, limit)

	// Execute direct query
	rows, err := t.db.QueryContext(ctx, directQuerySQL, queryArgs...)
	if err != nil {
		return nil, t.errorHandler.Wrap(err, "failed to execute search query", map[string]interface{}{
			"query":            query.Text,
			"embedding_length": len(query.Embedding),
		})
	}
	defer rows.Close()

	documents, scores, err = t.parseSearchResults(rows, query.Threshold, true)
	if err != nil {
		return nil, err
	}

	result := &domain.QueryResult{
		Documents: documents,
		Scores:    scores,
		Total:     len(documents),
		Query:     query,
	}

	t.logger.Debug("Direct search completed successfully", map[string]interface{}{
		"query":                query.Text,
		"results_count":        len(documents),
		"similarity_threshold": query.Threshold,
	})

	return result, nil
}

// Get retrieves a document by ID
func (t *TursoVectorStore) Get(ctx context.Context, id string) (*domain.Document, error) {
	query := fmt.Sprintf(`
		SELECT id, content, metadata, embedding, created_at, updated_at,
			   source, chunk_index, total_chunks
		FROM %s WHERE id = ?
	`, t.tableName)

	row := t.db.QueryRowContext(ctx, query, id)

	var doc domain.Document
	var metadataJSON string
	var embeddingBlob []byte

	err := row.Scan(
		&doc.ID,
		&doc.Content,
		&metadataJSON,
		&embeddingBlob,
		&doc.CreatedAt,
		&doc.UpdatedAt,
		&doc.Source,
		&doc.ChunkIndex,
		&doc.TotalChunks,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Document not found
		}
		return nil, t.errorHandler.Wrap(err, "failed to get document", map[string]interface{}{
			"document_id": id,
		})
	}

	// Deserialize metadata
	if err := json.Unmarshal([]byte(metadataJSON), &doc.Metadata); err != nil {
		return nil, t.errorHandler.Wrap(err, "failed to deserialize metadata", map[string]interface{}{
			"document_id": id,
		})
	}

	// For F32_BLOB, we'll need to extract using vector_extract if needed
	// For now, initialize empty embedding as it's stored in native format
	doc.Embedding = make([]float32, t.embedDimension)
	// TODO: Implement vector_extract if embedding field access is needed

	return &doc, nil
}

// Delete deletes documents by IDs
func (t *TursoVectorStore) Delete(ctx context.Context, ids []string) error {
	if len(ids) == 0 {
		return nil
	}

	placeholders := strings.Repeat("?,", len(ids))
	placeholders = placeholders[:len(placeholders)-1] // Remove trailing comma

	query := fmt.Sprintf("DELETE FROM %s WHERE id IN (%s)", t.tableName, placeholders)

	args := make([]interface{}, len(ids))
	for i, id := range ids {
		args[i] = id
	}

	result, err := t.db.ExecContext(ctx, query, args...)
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to delete documents", map[string]interface{}{
			"document_ids": ids,
		})
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to get affected rows", map[string]interface{}{
			"document_ids": ids,
		})
	}

	t.logger.Info("Successfully deleted documents", map[string]interface{}{
		"requested_count": len(ids),
		"deleted_count":   rowsAffected,
	})

	return nil
}

// Update updates a document
func (t *TursoVectorStore) Update(ctx context.Context, document domain.Document) error {
	if document.ID == "" {
		return t.errorHandler.New("document ID is required for update", map[string]interface{}{
			"document": document,
		})
	}

	document.UpdatedAt = time.Now()

	// Serialize metadata
	metadataJSON, err := json.Marshal(document.Metadata)
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to serialize metadata", map[string]interface{}{
			"document_id": document.ID,
		})
	}

	// Serialize embedding for vector32()
	embeddingJSON, err := json.Marshal(document.Embedding)
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to serialize embedding", map[string]interface{}{
			"document_id": document.ID,
		})
	}

	query := fmt.Sprintf(`
		UPDATE %s SET 
			content = ?, metadata = ?, embedding = vector32(?), updated_at = ?,
			source = ?, chunk_index = ?, total_chunks = ?
		WHERE id = ?
	`, t.tableName)

	result, err := t.db.ExecContext(ctx, query,
		document.Content,
		string(metadataJSON),
		string(embeddingJSON), // This will be passed to vector32() function
		document.UpdatedAt,
		document.Source,
		document.ChunkIndex,
		document.TotalChunks,
		document.ID,
	)
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to update document", map[string]interface{}{
			"document_id": document.ID,
		})
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return t.errorHandler.Wrap(err, "failed to get affected rows", map[string]interface{}{
			"document_id": document.ID,
		})
	}

	if rowsAffected == 0 {
		return t.errorHandler.New("document not found", map[string]interface{}{
			"document_id": document.ID,
		})
	}

	t.logger.Debug("Successfully updated document", map[string]interface{}{
		"document_id": document.ID,
	})

	return nil
}

// List lists all documents with optional filters
func (t *TursoVectorStore) List(ctx context.Context, filters map[string]interface{}, limit, offset int) ([]domain.Document, error) {
	whereClause, args := t.buildWhereClause(filters)

	if limit <= 0 {
		limit = 100 // Default limit
	}

	query := fmt.Sprintf(`
		SELECT id, content, metadata, embedding, created_at, updated_at,
			   source, chunk_index, total_chunks
		FROM %s
		%s
		ORDER BY created_at DESC
		LIMIT ? OFFSET ?
	`, t.tableName, whereClause)

	args = append(args, limit, offset)

	rows, err := t.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, t.errorHandler.Wrap(err, "failed to list documents", map[string]interface{}{
			"filters": filters,
			"limit":   limit,
			"offset":  offset,
		})
	}
	defer rows.Close()

	var documents []domain.Document

	for rows.Next() {
		var doc domain.Document
		var metadataJSON string
		var embeddingBlob []byte

		err := rows.Scan(
			&doc.ID,
			&doc.Content,
			&metadataJSON,
			&embeddingBlob,
			&doc.CreatedAt,
			&doc.UpdatedAt,
			&doc.Source,
			&doc.ChunkIndex,
			&doc.TotalChunks,
		)
		if err != nil {
			return nil, t.errorHandler.Wrap(err, "failed to scan document", map[string]interface{}{
				"document_id": doc.ID,
			})
		}

		// Deserialize metadata
		if err := json.Unmarshal([]byte(metadataJSON), &doc.Metadata); err != nil {
			return nil, t.errorHandler.Wrap(err, "failed to deserialize metadata", map[string]interface{}{
				"document_id": doc.ID,
			})
		}

		// Initialize embedding for F32_BLOB
		doc.Embedding = make([]float32, t.embedDimension)
		// TODO: Implement vector_extract if embedding field access is needed

		documents = append(documents, doc)
	}

	if err := rows.Err(); err != nil {
		return nil, t.errorHandler.Wrap(err, "error iterating documents", map[string]interface{}{
			"filters": filters,
		})
	}

	return documents, nil
}

// Stats returns collection statistics
func (t *TursoVectorStore) Stats(ctx context.Context) (map[string]interface{}, error) {
	stats := make(map[string]interface{})

	// Get total document count
	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM %s", t.tableName)
	var totalDocs int64
	if err := t.db.QueryRowContext(ctx, countQuery).Scan(&totalDocs); err != nil {
		return nil, t.errorHandler.Wrap(err, "failed to get document count", map[string]interface{}{
			"table_name": t.tableName,
		})
	}
	stats["total_documents"] = totalDocs

	// Get table size (approximate)
	sizeQuery := fmt.Sprintf("SELECT COUNT(*) * AVG(LENGTH(content)) FROM %s", t.tableName)
	var avgSize sql.NullFloat64
	if err := t.db.QueryRowContext(ctx, sizeQuery).Scan(&avgSize); err != nil {
		return nil, t.errorHandler.Wrap(err, "failed to get table size", map[string]interface{}{
			"table_name": t.tableName,
		})
	}
	if avgSize.Valid {
		stats["approximate_size_bytes"] = int64(avgSize.Float64)
	}

	stats["table_name"] = t.tableName
	stats["embed_dimension"] = t.embedDimension

	return stats, nil
}

// Close closes the vector store connection
func (t *TursoVectorStore) Close() error {
	if t.db != nil {
		if err := t.db.Close(); err != nil {
			return t.errorHandler.Wrap(err, "failed to close database connection", map[string]interface{}{
				"table_name": t.tableName,
			})
		}
	}

	t.logger.Info("TursoDB vector store closed successfully", map[string]interface{}{
		"table_name": t.tableName,
	})

	return nil
}

// Helper methods

// parseConfig parses the configuration map into TursoConfig struct
func (t *TursoVectorStore) parseConfig(config map[string]interface{}, tursoConfig *TursoConfig) error {
	if url, ok := config["url"].(string); ok {
		tursoConfig.URL = url
	}
	if authToken, ok := config["auth_token"].(string); ok {
		tursoConfig.AuthToken = authToken
	}
	if tableName, ok := config["table_name"].(string); ok {
		tursoConfig.TableName = tableName
	}
	if embedDimension, ok := config["embed_dimension"].(int); ok {
		tursoConfig.EmbedDimension = embedDimension
	}
	if createTable, ok := config["create_table"].(bool); ok {
		tursoConfig.CreateTable = createTable
	}

	return nil
}

// buildWhereClause builds a WHERE clause from filters
func (t *TursoVectorStore) buildWhereClause(filters map[string]interface{}) (string, []interface{}) {
	if len(filters) == 0 {
		return "", []interface{}{}
	}

	var conditions []string
	var args []interface{}

	for key, value := range filters {
		switch key {
		case "source":
			conditions = append(conditions, "source = ?")
			args = append(args, value)
		case "chunk_index":
			conditions = append(conditions, "chunk_index = ?")
			args = append(args, value)
		case "created_after":
			conditions = append(conditions, "created_at > ?")
			args = append(args, value)
		case "created_before":
			conditions = append(conditions, "created_at < ?")
			args = append(args, value)
		default:
			// For metadata filters, use JSON operations
			conditions = append(conditions, "json_extract(metadata, '$."+key+"') = ?")
			args = append(args, value)
		}
	}

	if len(conditions) == 0 {
		return "", []interface{}{}
	}

	return "WHERE " + strings.Join(conditions, " AND "), args
}

// parseSearchResults parses search results from SQL rows
func (t *TursoVectorStore) parseSearchResults(rows *sql.Rows, threshold float32, isDistance bool) ([]domain.Document, []float32, error) {
	var documents []domain.Document
	var scores []float32

	for rows.Next() {
		var doc domain.Document
		var metadataJSON string
		var embeddingBlob []byte
		var distance float64

		err := rows.Scan(
			&doc.ID,
			&doc.Content,
			&metadataJSON,
			&embeddingBlob,
			&doc.CreatedAt,
			&doc.UpdatedAt,
			&doc.Source,
			&doc.ChunkIndex,
			&doc.TotalChunks,
			&distance,
		)
		if err != nil {
			return nil, nil, t.errorHandler.Wrap(err, "failed to scan search result", map[string]interface{}{
				"document_id": doc.ID,
			})
		}

		// Deserialize metadata
		if err := json.Unmarshal([]byte(metadataJSON), &doc.Metadata); err != nil {
			return nil, nil, t.errorHandler.Wrap(err, "failed to deserialize metadata", map[string]interface{}{
				"document_id":   doc.ID,
				"metadata_json": metadataJSON,
			})
		}

		// Extract embedding from F32_BLOB using vector_extract
		// For now, we'll store the raw blob and extract when needed
		doc.Embedding = make([]float32, t.embedDimension)

		// TODO: Implement proper F32_BLOB extraction if needed for the embedding field

		// Convert distance to similarity score (cosine distance: 0=identical, 2=opposite)
		// So similarity = 1 - (distance/2) to get 0-1 range
		var score float32
		if isDistance {
			score = 1.0 - float32(distance)/2.0
		} else {
			score = float32(distance)
		}

		// Apply similarity threshold filter
		if threshold > 0 && score < threshold {
			continue
		}

		documents = append(documents, doc)
		scores = append(scores, score)
	}

	if err := rows.Err(); err != nil {
		return nil, nil, t.errorHandler.Wrap(err, "error iterating search results", nil)
	}

	return documents, scores, nil
}
