package providers

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	_ "github.com/tursodatabase/libsql-client-go/libsql"

	"github.com/ZanzyTHEbar/genkithandler/pkg/domain"
)

const (
	testDbURL          = "file::memory:?cache=shared"
	testDbAuthToken    = "" // No auth token needed for in-memory SQLite
	testEmbeddingTable = "test_vector_embeddings"
	testEmbeddingDim   = 3 // Example dimension
)

// Helper function to create a new TursoVectorStore for testing
func newTestTursoDBProvider(t *testing.T) (*TursoVectorStore, func()) { // Changed TursoDBProvider to TursoVectorStore
	// Use an in-memory SQLite database for testing
	// The `cache=shared` is important to allow multiple connections to the same in-memory DB.
	// A unique name for each in-memory DB to avoid interference between tests if run in parallel.
	dbName := fmt.Sprintf("file:%s?cache=shared&mode=memory", uuid.New().String())

	db, err := sql.Open("libsql", dbName)
	assert.NoError(t, err, "Failed to open in-memory SQLite DB for testing")

	provider := &TursoVectorStore{ // Changed TursoDBProvider to TursoVectorStore
		db:             db,
		tableName:      testEmbeddingTable, // Corrected field name
		embedDimension: testEmbeddingDim,   // Corrected field name
	}

	// Create the embeddings table
	err = provider.createVectorTable(context.Background()) // Corrected method call
	assert.NoError(t, err, "Failed to create embeddings table for testing")

	// Teardown function to close the database
	teardown := func() {
		err := db.Close()
		assert.NoError(t, err, "Failed to close test database")
	}

	return provider, teardown
}

func TestMain(m *testing.M) {
	// TODO: Setup code, if any, can go here.
	// For example, setting up a global in-memory database if preferred over per-test DBs.
	// However, for TursoVectorStore, it's cleaner to set up the DB per test or per suite
	// to ensure test isolation, especially with schema creation.

	exitVal := m.Run()

	// TODO: Teardown code, if any, can go here.
	os.Exit(exitVal)
}

func TestTursoDBProvider_ensureTableExists(t *testing.T) { // Changed TursoDBProvider to TursoVectorStore
	dbName := fmt.Sprintf("file:ensure_table_exists_%s?cache=shared&mode=memory", uuid.New().String())
	db, err := sql.Open("libsql", dbName)
	assert.NoError(t, err)
	defer db.Close()

	provider := &TursoVectorStore{ // Changed TursoDBProvider to TursoVectorStore
		db:             db,
		tableName:      "test_ensure_table", // Corrected field name
		embedDimension: 3,                   // Corrected field name
	}

	err = provider.createVectorTable(context.Background()) // Corrected method call
	assert.NoError(t, err, "ensureTableExists should not return an error on first call")

	// Verify table exists
	var tableName string
	query := fmt.Sprintf("SELECT name FROM sqlite_master WHERE type='table' AND name='%s';", provider.tableName)
	err = db.QueryRowContext(context.Background(), query).Scan(&tableName)
	assert.NoError(t, err, "Failed to query for table name")
	assert.Equal(t, provider.tableName, tableName, "Table should exist after ensureTableExists")

	// Call again to ensure idempotency
	err = provider.createVectorTable(context.Background()) // Corrected method call
	assert.NoError(t, err, "ensureTableExists should not return an error on subsequent calls")
}

func TestTursoDBProvider_StoreEmbeddings_And_RetrieveEmbeddings(t *testing.T) { // Function name kept for consistency, refers to TursoVectorStore now
	provider, teardown := newTestTursoDBProvider(t)
	defer teardown()

	ctx := context.Background()

	testCases := []struct {
		name                string
		embeddingsToStore   []domain.Document
		queryEmbedding      domain.Document
		k                   int
		expectedRetrieved   int
		expectStoreError    bool
		expectRetrieveError bool
	}{
		{
			name: "Store and retrieve single embedding",
			embeddingsToStore: []domain.Document{
				{ID: "doc1", Embedding: []float32{0.1, 0.2, 0.3}, Metadata: map[string]interface{}{"source": "doc1"}},
			},
			queryEmbedding:    domain.Document{Embedding: []float32{0.1, 0.2, 0.3}},
			k:                 1,
			expectedRetrieved: 1,
		},
		{
			name: "Store multiple, retrieve top K",
			embeddingsToStore: []domain.Document{
				{ID: "docA", Embedding: []float32{1.0, 1.0, 1.0}, Metadata: map[string]interface{}{"source": "A"}},
				{ID: "docB", Embedding: []float32{0.1, 0.2, 0.3}, Metadata: map[string]interface{}{"source": "B"}}, // Closest
				{ID: "docC", Embedding: []float32{0.5, 0.5, 0.5}, Metadata: map[string]interface{}{"source": "C"}},
			},
			queryEmbedding:    domain.Document{Embedding: []float32{0.1, 0.1, 0.1}},
			k:                 2,
			expectedRetrieved: 2,
		},
		{
			name: "Retrieve fewer than K when not enough documents",
			embeddingsToStore: []domain.Document{
				{ID: "docX", Embedding: []float32{0.7, 0.8, 0.9}},
			},
			queryEmbedding:    domain.Document{Embedding: []float32{0.6, 0.7, 0.8}},
			k:                 5,
			expectedRetrieved: 1,
		},
		{
			name: "Retrieve with no matches",
			embeddingsToStore: []domain.Document{
				{ID: "docY", Embedding: []float32{10.0, 20.0, 30.0}},
			},
			queryEmbedding:    domain.Document{Embedding: []float32{0.1, 0.2, 0.3}}, // Very different
			k:                 1,
			expectedRetrieved: 1, // Changed from 0 to 1, as vector_top_k will return the closest even if far
		},
		{
			name:              "Store empty embeddings slice",
			embeddingsToStore: []domain.Document{},
			queryEmbedding:    domain.Document{Embedding: []float32{0.1, 0.2, 0.3}},
			k:                 1,
			expectedRetrieved: 0,
		},
		{
			name: "Store embedding with incorrect dimension",
			embeddingsToStore: []domain.Document{
				{ID: "doc_wrong_dim", Embedding: []float32{0.1, 0.2}}, // Dim 2, expected 3
			},
			// The vector32 function in Turso is flexible with input JSON array length.
			// The F32_BLOB(dim) in table creation enforces dimension for the VSS index.
			// Storing a vector of a different dimension than `embedDimension` will likely cause an error
			// during the INSERT if the VSS index tries to process it, or during SEARCH.
			expectStoreError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) { // Corrected t *testing.t to t *testing.T
			// Clear the table for each test case to ensure isolation
			_, err := provider.db.ExecContext(ctx, fmt.Sprintf("DELETE FROM %s;", provider.tableName))
			assert.NoError(t, err, "Failed to clear embeddings table before test")

			err = provider.Store(ctx, tc.embeddingsToStore) // Corrected method call
			if tc.expectStoreError {
				assert.Error(t, err, "Expected an error during StoreEmbeddings")
				return
			}
			assert.NoError(t, err, "StoreEmbeddings failed unexpectedly")

			if len(tc.embeddingsToStore) == 0 && !tc.expectStoreError {
				retrieved, err := provider.Search(ctx, domain.Query{Embedding: tc.queryEmbedding.Embedding, Limit: tc.k}) // Corrected method call and query construction
				assert.NoError(t, err, "RetrieveEmbeddings failed unexpectedly after storing empty slice")
				assert.Len(t, retrieved.Documents, 0, "Expected 0 embeddings retrieved after storing empty slice")
				return
			}

			if !tc.expectStoreError && len(tc.embeddingsToStore) > 0 {
				retrievedResult, err := provider.Search(ctx, domain.Query{Embedding: tc.queryEmbedding.Embedding, Limit: tc.k}) // Corrected method call and query construction
				if tc.expectRetrieveError {
					assert.Error(t, err, "Expected an error during RetrieveEmbeddings")
					return
				}
				assert.NoError(t, err, "RetrieveEmbeddings failed unexpectedly")
				assert.Len(t, retrievedResult.Documents, tc.expectedRetrieved, "Retrieved incorrect number of embeddings")

				if tc.expectedRetrieved > 0 && len(retrievedResult.Documents) > 0 {
					storedIDs := make(map[string]bool)
					for _, emb := range tc.embeddingsToStore {
						storedIDs[emb.ID] = true
					}
					for _, retEmb := range retrievedResult.Documents {
						assert.True(t, storedIDs[retEmb.ID], fmt.Sprintf("Retrieved document ID %s was not in the stored documents", retEmb.ID))
						if tc.name == "Store and retrieve single embedding" {
							assert.Equal(t, tc.embeddingsToStore[0].Metadata, retEmb.Metadata)
						}
					}
				}
			} else if !tc.expectStoreError && len(tc.embeddingsToStore) == 0 {
				retrievedResult, err := provider.Search(ctx, domain.Query{Embedding: tc.queryEmbedding.Embedding, Limit: tc.k}) // Corrected method call and query construction
				assert.NoError(t, err, "RetrieveEmbeddings failed unexpectedly when nothing was stored")
				assert.Len(t, retrievedResult.Documents, 0, "Expected 0 embeddings when nothing was stored")
			}
		})
	}
}

func TestTursoDBProvider_StoreEmbeddings_ErrorHandling(t *testing.T) { // Function name kept for consistency
	provider, teardown := newTestTursoDBProvider(t)
	defer teardown()

	ctx := context.Background()

	t.Run("Store embedding with nil vector", func(t *testing.T) {
		embeddings := []domain.Document{
			{ID: "nil_vector_doc", Embedding: nil, Metadata: map[string]interface{}{"source": "test"}},
		}
		err := provider.Store(ctx, embeddings) // Corrected method call
		assert.Error(t, err, "Storing an embedding with a nil vector should ideally result in an error from the DB")
	})

	t.Run("Store embedding with empty vector", func(t *testing.T) {
		embeddings := []domain.Document{
			{ID: "empty_vector_doc", Embedding: []float32{}, Metadata: map[string]interface{}{"source": "test"}},
		}
		err := provider.Store(ctx, embeddings) // Corrected method call
		assert.Error(t, err, "Storing an embedding with an empty vector should ideally result in an error from the DB")
	})

	t.Run("Store embedding with DB error", func(t *testing.T) {
		dbName := fmt.Sprintf("file:db_error_store_%s?cache=shared&mode=memory", uuid.New().String())
		db, err := sql.Open("libsql", dbName)
		assert.NoError(t, err)

		errorProvider := &TursoVectorStore{ // Changed TursoDBProvider to TursoVectorStore
			db:             db,
			tableName:      testEmbeddingTable, // Corrected field name
			embedDimension: testEmbeddingDim,   // Corrected field name
		}
		err = errorProvider.createVectorTable(context.Background()) // Corrected method call
		assert.NoError(t, err)

		db.Close()

		embeddings := []domain.Document{
			{ID: "doc_db_error", Embedding: []float32{0.1, 0.2, 0.3}},
		}
		err = errorProvider.Store(ctx, embeddings) // Corrected method call
		assert.Error(t, err, "Expected an error when storing embeddings with a closed DB connection")
	})
}

func TestTursoDBProvider_RetrieveEmbeddings_ErrorHandling(t *testing.T) { // Function name kept for consistency
	provider, teardown := newTestTursoDBProvider(t)
	defer teardown()

	ctx := context.Background()

	validEmbedding := []domain.Document{
		{ID: "valid_doc", Embedding: []float32{0.5, 0.5, 0.5}, Metadata: map[string]interface{}{"source": "valid"}},
	}
	err := provider.Store(ctx, validEmbedding) // Corrected method call
	assert.NoError(t, err, "Setup: Failed to store valid embedding")

	t.Run("Retrieve with K=0", func(t *testing.T) {
		queryEmbedding := domain.Document{Embedding: []float32{0.1, 0.2, 0.3}}
		retrieved, err := provider.Search(ctx, domain.Query{Embedding: queryEmbedding.Embedding, Limit: 0}) // Corrected method call
		assert.NoError(t, err, "RetrieveEmbeddings with K=0 should not error")
		assert.Len(t, retrieved.Documents, 0, "RetrieveEmbeddings with K=0 should return 0 results")
	})

	t.Run("Retrieve with K<0", func(t *testing.T) {
		queryEmbedding := domain.Document{Embedding: []float32{0.1, 0.2, 0.3}}
		retrieved, err := provider.Search(ctx, domain.Query{Embedding: queryEmbedding.Embedding, Limit: -1}) // Corrected method call
		assert.NoError(t, err, "RetrieveEmbeddings with K<0 should not error")
		assert.Len(t, retrieved.Documents, 0, "RetrieveEmbeddings with K<0 should return 0 results (SQLite behavior)")
	})

	t.Run("Retrieve with nil query vector", func(t *testing.T) {
		queryEmbedding := domain.Document{Embedding: nil}
		_, err := provider.Search(ctx, domain.Query{Embedding: queryEmbedding.Embedding, Limit: 1}) // Corrected method call
		assert.Error(t, err, "RetrieveEmbeddings with nil query vector should error")
	})

	t.Run("Retrieve with empty query vector", func(t *testing.T) { // Corrected t *testing.t to t *testing.T
		queryEmbedding := domain.Document{Embedding: []float32{}}
		_, err := provider.Search(ctx, domain.Query{Embedding: queryEmbedding.Embedding, Limit: 1}) // Corrected method call
		assert.Error(t, err, "RetrieveEmbeddings with empty query vector should error")
	})

	t.Run("Retrieve embedding with DB error", func(t *testing.T) {
		dbName := fmt.Sprintf("file:db_error_retrieve_%s?cache=shared&mode=memory", uuid.New().String())
		db, err := sql.Open("libsql", dbName)
		assert.NoError(t, err)

		errorProvider := &TursoVectorStore{ // Changed TursoDBProvider to TursoVectorStore
			db:             db,
			tableName:      testEmbeddingTable, // Corrected field name
			embedDimension: testEmbeddingDim,   // Corrected field name
		}
		err = errorProvider.createVectorTable(context.Background()) // Corrected method call
		assert.NoError(t, err)

		setupEmbedding := []domain.Document{{ID: "setup", Embedding: []float32{0.1, 0.1, 0.1}}}
		err = errorProvider.Store(ctx, setupEmbedding) // Corrected method call
		assert.NoError(t, err)

		db.Close()

		queryEmbedding := domain.Document{Embedding: []float32{0.1, 0.2, 0.3}}
		_, err = errorProvider.Search(ctx, domain.Query{Embedding: queryEmbedding.Embedding, Limit: 1}) // Corrected method call
		assert.Error(t, err, "Expected an error when retrieving embeddings with a closed DB connection")
	})
}

func TestTursoDBProvider_EmbeddingDimensionMismatch(t *testing.T) { // Function name kept for consistency
	provider, teardown := newTestTursoDBProvider(t)
	defer teardown()
	ctx := context.Background()

	correctDimEmbedding := domain.Document{
		ID: "correct_dim_doc", Embedding: []float32{0.1, 0.2, 0.3}, // Dim 3
	}
	err := provider.Store(ctx, []domain.Document{correctDimEmbedding}) // Corrected method call
	assert.NoError(t, err, "Failed to store embedding with correct dimension")

	t.Run("Query with different dimension vector", func(t *testing.T) {
		wrongDimQueryEmbedding := domain.Document{
			Embedding: []float32{0.1, 0.2, 0.3, 0.4}, // Dim 4
		}
		_, err := provider.Search(ctx, domain.Query{Embedding: wrongDimQueryEmbedding.Embedding, Limit: 1}) // Corrected method call
		assert.Error(t, err, "Expected an error when query vector dimension mismatches stored vector dimension")
	})

	t.Run("Store and retrieve vector with dimension different from table's VSS index hint", func(t *testing.T) {
		_, err := provider.db.ExecContext(ctx, fmt.Sprintf("DELETE FROM %s;", provider.tableName))
		assert.NoError(t, err)

		diffDimEmbedding := domain.Document{
			ID: "diff_dim_doc", Embedding: []float32{1.0, 2.0, 3.0, 4.0}, // Dim 4
		}
		err = provider.Store(ctx, []domain.Document{diffDimEmbedding}) // Corrected method call
		// This should error because the table is created with F32_BLOB(embedDimension) which is 3 for tests.
		// Storing a 4-dim vector into a column expecting 3-dim vectors via VSS should fail.
		assert.Error(t, err, "Storing a vector with dimension different from VSS index hint should error")

		// If Store did not error (which it should have), then Search might also error or give weird results.
		// However, the primary check is that Store should fail first.
		// queryVec := domain.Document{Embedding: []float32{1.1, 2.1, 3.1, 4.1}}
		// _, errSearch := provider.Search(ctx, domain.Query{Embedding: queryVec.Embedding, Limit: 1})
		// assert.Error(t, errSearch, "Searching with a vector of mismatched dimension should also error or be handled")
	})

	providerFresh, teardownFresh := newTestTursoDBProvider(t)
	defer teardownFresh()

	mismatchedDimForStore := domain.Document{
		ID: "mismatched_store", Embedding: []float32{1.0, 2.0, 3.0, 4.0}, // Dim 4
	}
	errFreshStore := providerFresh.Store(ctx, []domain.Document{mismatchedDimForStore}) // Corrected method call
	assert.Error(t, errFreshStore, "Storing a vector whose dimension mismatches the VSS table schema should error")
}
