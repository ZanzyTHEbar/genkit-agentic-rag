package plugin

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// AgenticRAGProcessor implements the core agentic RAG flow
type AgenticRAGProcessor struct {
	config *AgenticRAGConfig
}

// NewAgenticRAGProcessor creates a new processor with the given configuration
func NewAgenticRAGProcessor(config *AgenticRAGConfig) *AgenticRAGProcessor {
	if config == nil {
		config = DefaultConfig()
	}
	return &AgenticRAGProcessor{
		config: config,
	}
}

// DefaultConfig returns a default configuration
func DefaultConfig() *AgenticRAGConfig {
	return &AgenticRAGConfig{
		ModelName: "googleai/gemini-2.5-flash", // Default model name - DO NOT CHANGE
		Processing: ProcessingConfig{
			DefaultChunkSize:      1000,
			DefaultMaxChunks:      20,
			DefaultRecursiveDepth: 3,
			RespectSentences:      true,
		},
		KnowledgeGraph: KnowledgeGraphConfig{
			Enabled:                true,
			EntityTypes:            []string{"PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "TECHNOLOGY", "EVENT"},
			RelationTypes:          []string{"WORKS_FOR", "LOCATED_IN", "FOUNDED", "DEVELOPS", "USES", "RELATED_TO"},
			MinConfidenceThreshold: 0.7,
		},
		FactVerification: FactVerificationConfig{
			Enabled:            true,
			RequireEvidence:    true,
			MinConfidenceScore: 0.7,
		},
		Prompts: PromptsConfig{
			Directory:                 "./prompts",
			RelevanceScoringPrompt:    "relevance_scoring",
			ResponseGenerationPrompt:  "response_generation",
			KnowledgeExtractionPrompt: "knowledge_extraction",
			FactVerificationPrompt:    "fact_verification",
			Variants:                  make(map[string]string),
			CustomHelpers:             true,
		},
	}
}

// initializePrompts sets up the prompt system with custom helpers
func (p *AgenticRAGProcessor) initializePrompts(ctx context.Context) error {
	if p.config.Genkit == nil {
		return fmt.Errorf("GenKit instance not provided in config")
	}

	g := p.config.Genkit

	// Register custom helpers for prompt templates
	if p.config.Prompts.CustomHelpers {
		// Helper to create arrays in templates
		genkit.DefineHelper(g, "array", func(items ...interface{}) []interface{} {
			return items
		})

		// Helper to format confidence scores
		genkit.DefineHelper(g, "confidence", func(score float64) string {
			return fmt.Sprintf("%.2f", score)
		})

		// Helper to truncate text with ellipsis
		genkit.DefineHelper(g, "truncate", func(text string, length int) string {
			if len(text) <= length {
				return text
			}
			return text[:length] + "..."
		})

		// Helper to join array elements
		genkit.DefineHelper(g, "join", func(items []string, separator string) string {
			return strings.Join(items, separator)
		})

		// Helper to format entity types
		genkit.DefineHelper(g, "entityTypes", func(types []string) string {
			if len(types) == 0 {
				return ""
			}
			if len(types) == 1 {
				return types[0]
			}
			return strings.Join(types[:len(types)-1], ", ") + " and " + types[len(types)-1]
		})
	}

	return nil
}

// Process executes the agentic RAG flow according to the specification
func (p *AgenticRAGProcessor) Process(ctx context.Context, request AgenticRAGRequest) (*AgenticRAGResponse, error) {
	startTime := time.Now()

	// Set default options
	if request.Options.MaxChunks == 0 {
		request.Options.MaxChunks = p.config.Processing.DefaultMaxChunks
	}
	if request.Options.RecursiveDepth == 0 {
		request.Options.RecursiveDepth = p.config.Processing.DefaultRecursiveDepth
	}
	if request.Options.Temperature == 0 {
		request.Options.Temperature = 0.7 // Default temperature
	}

	// Step 1: Load documents into context window
	documents, err := p.loadDocuments(ctx, request.Documents)
	if err != nil {
		return nil, fmt.Errorf("failed to load documents: %w", err)
	}

	// Step 2: Chunk documents into initial chunks (respecting sentence boundaries)
	allChunks := make([]DocumentChunk, 0)
	for _, doc := range documents {
		chunks, err := p.chunkDocument(ctx, doc, request.Options.MaxChunks)
		if err != nil {
			return nil, fmt.Errorf("failed to chunk document %s: %w", doc.ID, err)
		}
		allChunks = append(allChunks, chunks...)
	}

	// Step 3: Prompt model to identify relevant chunks
	relevantChunks, err := p.identifyRelevantChunks(ctx, request.Query, allChunks)
	if err != nil {
		return nil, fmt.Errorf("failed to identify relevant chunks: %w", err)
	}

	// Step 4 & 5: Recursively drill down into selected chunks
	finalChunks, recursiveLevels, err := p.recursivelyRefineChunks(ctx, request.Query, relevantChunks, request.Options.RecursiveDepth)
	if err != nil {
		return nil, fmt.Errorf("failed to recursively refine chunks: %w", err)
	}

	// Step 6: Generate response based on retrieved information
	answer, tokenCount, err := p.generateResponse(ctx, request.Query, finalChunks, request.Options)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}

	// Step 7: Build knowledge graph if enabled
	var knowledgeGraph *KnowledgeGraph
	if request.Options.EnableKnowledgeGraph && p.config.KnowledgeGraph.Enabled {
		knowledgeGraph, err = p.buildKnowledgeGraph(ctx, finalChunks)
		if err != nil {
			return nil, fmt.Errorf("failed to build knowledge graph: %w", err)
		}
	}

	// Step 8: Verify answer for factual accuracy if enabled
	var factVerification *FactVerification
	if request.Options.EnableFactVerification {
		factVerification, err = p.verifyFacts(ctx, answer, finalChunks)
		if err != nil {
			return nil, fmt.Errorf("failed to verify facts: %w", err)
		}
	}

	// Convert chunks to processed chunks format
	processedChunks := make([]ProcessedChunk, len(finalChunks))
	for i, chunk := range finalChunks {
		processedChunks[i] = ProcessedChunk{
			Chunk: chunk,
			// Entities and Relations will be populated during knowledge graph building
		}
	}

	return &AgenticRAGResponse{
		Answer:           answer,
		RelevantChunks:   processedChunks,
		KnowledgeGraph:   knowledgeGraph,
		FactVerification: factVerification,
		ProcessingMetadata: ProcessingMetadata{
			ProcessingTime:  time.Since(startTime),
			ChunksProcessed: len(allChunks),
			RecursiveLevels: recursiveLevels,
			ModelCalls:      1 + recursiveLevels + 1, // identification + recursive calls + generation
			TokensUsed:      tokenCount,
		},
	}, nil
}

// loadDocuments loads documents from various sources
func (p *AgenticRAGProcessor) loadDocuments(ctx context.Context, sources []string) ([]Document, error) {
	documents := make([]Document, 0, len(sources))

	for i, source := range sources {
		doc := Document{
			ID:      fmt.Sprintf("doc_%d", i),
			Content: source, // For MVP, treat as raw text
			Source:  source,
			Metadata: map[string]interface{}{
				"loaded_at": time.Now(),
			},
		}
		documents = append(documents, doc)
	}

	return documents, nil
}

// chunkDocument breaks a document into chunks respecting sentence boundaries
func (p *AgenticRAGProcessor) chunkDocument(ctx context.Context, doc Document, maxChunks int) ([]DocumentChunk, error) {
	chunkSize := p.config.Processing.DefaultChunkSize
	content := doc.Content

	// Simple sentence-aware chunking
	sentences := p.splitIntoSentences(content)
	chunks := make([]DocumentChunk, 0)

	currentChunk := ""
	currentStart := 0
	chunkIndex := 0

	for _, sentence := range sentences {
		// If adding this sentence would exceed chunk size, finalize current chunk
		if len(currentChunk)+len(sentence) > chunkSize && currentChunk != "" {
			chunk := DocumentChunk{
				ID:         fmt.Sprintf("%s_chunk_%d", doc.ID, chunkIndex),
				Content:    strings.TrimSpace(currentChunk),
				DocumentID: doc.ID,
				ChunkIndex: chunkIndex,
				StartIndex: currentStart,
				EndIndex:   currentStart + len(currentChunk),
			}
			chunks = append(chunks, chunk)

			// Start new chunk
			chunkIndex++
			currentStart = currentStart + len(currentChunk)
			currentChunk = sentence + " "

			// Stop if we've reached max chunks
			if len(chunks) >= maxChunks {
				break
			}
		} else {
			currentChunk += sentence + " "
		}
	}

	// Add final chunk if it has content
	if currentChunk != "" && len(chunks) < maxChunks {
		chunk := DocumentChunk{
			ID:         fmt.Sprintf("%s_chunk_%d", doc.ID, chunkIndex),
			Content:    strings.TrimSpace(currentChunk),
			DocumentID: doc.ID,
			ChunkIndex: chunkIndex,
			StartIndex: currentStart,
			EndIndex:   currentStart + len(currentChunk),
		}
		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

// splitIntoSentences splits text into sentences using simple regex
func (p *AgenticRAGProcessor) splitIntoSentences(text string) []string {
	// Simple sentence splitting regex
	sentenceRegex := regexp.MustCompile(`[.!?]+\s+`)
	sentences := sentenceRegex.Split(text, -1)

	// Filter out empty sentences
	result := make([]string, 0, len(sentences))
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence != "" {
			result = append(result, sentence)
		}
	}

	return result
}

// identifyRelevantChunks uses LLM to identify which chunks are most relevant to the query
func (p *AgenticRAGProcessor) identifyRelevantChunks(ctx context.Context, query string, chunks []DocumentChunk) ([]DocumentChunk, error) {
	if len(chunks) == 0 {
		return chunks, nil
	}

	// Initialize prompts if not done already
	if err := p.initializePrompts(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize prompts: %w", err)
	}

	// Prepare chunk content for prompt
	chunkTexts := make([]string, len(chunks))
	for i, chunk := range chunks {
		chunkTexts[i] = chunk.Content
	}

	// Get the prompt variant to use (default or configured variant)
	promptName := p.config.Prompts.RelevanceScoringPrompt
	if variant, exists := p.config.Prompts.Variants["relevance_scoring"]; exists {
		promptName = fmt.Sprintf("%s.%s", promptName, variant)
	}

	// Lookup the dotprompt
	relevancePrompt := genkit.LookupPrompt(p.config.Genkit, promptName)
	if relevancePrompt == nil {
		// Fallback to hardcoded prompt if dotprompt not found
		return p.identifyRelevantChunksFallback(ctx, query, chunks)
	}

	// Execute the prompt with proper input
	response, err := relevancePrompt.Execute(ctx,
		ai.WithInput(map[string]any{
			"query":      query,
			"chunks":     chunkTexts,
			"max_chunks": p.config.Processing.DefaultMaxChunks,
		}),
	)
	if err != nil {
		// Fallback to simple scoring if LLM fails
		return p.fallbackRelevanceScoring(query, chunks), nil
	}

	// Parse the structured response
	var responseData map[string]any
	if err := response.Output(&responseData); err != nil {
		// Fallback if parsing fails
		return p.fallbackRelevanceScoring(query, chunks), nil
	}

	// Extract chunk scores from response
	return p.parseRelevanceResponseData(responseData, chunks)
}

// identifyRelevantChunksFallback provides a fallback when dotprompt is not available
func (p *AgenticRAGProcessor) identifyRelevantChunksFallback(ctx context.Context, query string, chunks []DocumentChunk) ([]DocumentChunk, error) {
	// Create a prompt for the LLM to score chunk relevance
	prompt := fmt.Sprintf(`You are an expert at analyzing document relevance. Given a query and a list of document chunks, 
score each chunk from 0.0 to 1.0 based on how relevant it is to answering the query.

Query: "%s"

Document Chunks:
`, query)

	for i, chunk := range chunks {
		prompt += fmt.Sprintf("\n[%d] %s", i, chunk.Content)
	}

	prompt += `

Respond with a JSON array where each element has "index" (0-based chunk index) and "score" (0.0-1.0 relevance score).
Only include chunks with score > 0.3. Order by relevance score (highest first).

Example: [{"index": 2, "score": 0.9}, {"index": 0, "score": 0.7}]`

	// Use genkit.Generate to get LLM response
	model := p.config.Model
	var response *ai.ModelResponse
	var err error

	if model == nil {
		// Use model by name if no model instance available
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModelName(p.config.ModelName),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     0.1, // Low temperature for consistent scoring
				MaxOutputTokens: 1000,
			}),
		)
	} else {
		// Use model instance
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModel(model),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     0.1, // Low temperature for consistent scoring
				MaxOutputTokens: 1000,
			}),
		)
	}

	if err != nil {
		// Final fallback to simple keyword matching
		return p.fallbackRelevanceScoring(query, chunks), nil
	}

	responseText := response.Text()
	return p.parseRelevanceResponse(responseText, chunks)
}

// parseRelevanceResponseData parses structured response data from dotprompt
func (p *AgenticRAGProcessor) parseRelevanceResponseData(responseData map[string]any, chunks []DocumentChunk) ([]DocumentChunk, error) {
	chunksData, ok := responseData["chunks"]
	if !ok {
		return p.fallbackRelevanceScoring("", chunks), nil
	}

	chunksArray, ok := chunksData.([]any)
	if !ok {
		return p.fallbackRelevanceScoring("", chunks), nil
	}

	relevantChunks := make([]DocumentChunk, 0)

	for _, chunkData := range chunksArray {
		chunkMap, ok := chunkData.(map[string]any)
		if !ok {
			continue
		}

		indexFloat, ok := chunkMap["chunk_index"].(float64)
		if !ok {
			continue
		}
		index := int(indexFloat)

		scoreFloat, ok := chunkMap["relevance_score"].(float64)
		if !ok {
			continue
		}

		// Validate index and score
		if index >= 0 && index < len(chunks) && scoreFloat >= 0.3 {
			chunk := chunks[index]
			chunk.RelevanceScore = scoreFloat
			relevantChunks = append(relevantChunks, chunk)
		}
	}

	// Sort by relevance score (highest first)
	sort.Slice(relevantChunks, func(i, j int) bool {
		return relevantChunks[i].RelevanceScore > relevantChunks[j].RelevanceScore
	})

	return relevantChunks, nil
}

// parseRelevanceResponse parses the LLM response for relevance scores
func (p *AgenticRAGProcessor) parseRelevanceResponse(responseText string, chunks []DocumentChunk) ([]DocumentChunk, error) {
	// Parse the LLM response
	var relevanceScores []struct {
		Index int     `json:"index"`
		Score float64 `json:"score"`
	}

	if err := json.Unmarshal([]byte(responseText), &relevanceScores); err != nil {
		// Fallback if JSON parsing fails
		return p.fallbackRelevanceScoring("", chunks), nil
	}

	// Apply scores and filter relevant chunks
	//
	relevantChunks := make([]DocumentChunk, 0)
	for _, score := range relevanceScores {
		if score.Index >= 0 && score.Index < len(chunks) && score.Score > 0.3 {
			chunk := chunks[score.Index]
			chunk.RelevanceScore = score.Score
			relevantChunks = append(relevantChunks, chunk)
		}
	}

	// Sort by relevance score (highest first)
	sort.Slice(relevantChunks, func(i, j int) bool {
		return relevantChunks[i].RelevanceScore > relevantChunks[j].RelevanceScore
	})

	// Return top chunks (up to half for recursive refinement)
	maxRelevant := len(chunks) / 2
	if maxRelevant > len(relevantChunks) {
		maxRelevant = len(relevantChunks)
	}

	return relevantChunks[:maxRelevant], nil
}

// fallbackRelevanceScoring provides simple keyword-based relevance scoring as a fallback
func (p *AgenticRAGProcessor) fallbackRelevanceScoring(query string, chunks []DocumentChunk) []DocumentChunk {
	relevantChunks := make([]DocumentChunk, 0)

	for _, chunk := range chunks {
		score := p.calculateRelevanceScore(query, chunk.Content)
		if score > 0.3 { // Simple threshold
			chunk.RelevanceScore = score
			relevantChunks = append(relevantChunks, chunk)
		}
	}

	// Sort by relevance score (highest first)
	sort.Slice(relevantChunks, func(i, j int) bool {
		return relevantChunks[i].RelevanceScore > relevantChunks[j].RelevanceScore
	})

	// Return top chunks (up to half for recursive refinement)
	maxRelevant := len(chunks) / 2
	if maxRelevant > len(relevantChunks) {
		maxRelevant = len(relevantChunks)
	}

	return relevantChunks[:maxRelevant]
}

// calculateRelevanceScore calculates a simple relevance score
func (p *AgenticRAGProcessor) calculateRelevanceScore(query, content string) float64 {
	queryWords := strings.Fields(strings.ToLower(query))
	contentLower := strings.ToLower(content)

	matches := 0
	for _, word := range queryWords {
		if strings.Contains(contentLower, word) {
			matches++
		}
	}

	return float64(matches) / float64(len(queryWords))
}

// recursivelyRefineChunks recursively drills down into chunks for more granular information
func (p *AgenticRAGProcessor) recursivelyRefineChunks(ctx context.Context, query string, chunks []DocumentChunk, maxDepth int) ([]DocumentChunk, int, error) {
	if maxDepth <= 0 || len(chunks) == 0 {
		return chunks, 0, nil
	}

	// For each chunk, break it down further if it's still too large
	refinedChunks := make([]DocumentChunk, 0)
	currentDepth := 0

	for _, chunk := range chunks {
		// If chunk is large enough, break it down further
		if len(chunk.Content) > 200 { // Paragraph-level threshold
			subChunks := p.breakdownChunk(chunk)

			// Recursively process sub-chunks
			if len(subChunks) > 1 {
				relevantSubChunks, _ := p.identifyRelevantChunks(ctx, query, subChunks)
				if len(relevantSubChunks) > 0 {
					furtherRefined, depth, _ := p.recursivelyRefineChunks(ctx, query, relevantSubChunks, maxDepth-1)
					refinedChunks = append(refinedChunks, furtherRefined...)
					if depth+1 > currentDepth {
						currentDepth = depth + 1
					}
					continue
				}
			}
		}

		// If we can't break it down further or it's already small, keep as is
		refinedChunks = append(refinedChunks, chunk)
	}

	return refinedChunks, currentDepth, nil
}

// breakdownChunk breaks a chunk into smaller sub-chunks
func (p *AgenticRAGProcessor) breakdownChunk(chunk DocumentChunk) []DocumentChunk {
	// Break into sentences for paragraph-level content
	sentences := p.splitIntoSentences(chunk.Content)

	if len(sentences) <= 1 {
		return []DocumentChunk{chunk}
	}

	subChunks := make([]DocumentChunk, 0, len(sentences))
	for idx, sentence := range sentences {
		subChunk := DocumentChunk{
			ID:         fmt.Sprintf("%s_sub_%d", chunk.ID, idx),
			Content:    sentence,
			DocumentID: chunk.DocumentID,
			ChunkIndex: chunk.ChunkIndex*100 + idx, // Hierarchical indexing
			StartIndex: chunk.StartIndex,           // Simplified for MVP
			EndIndex:   chunk.EndIndex,             // Simplified for MVP
		}
		subChunks = append(subChunks, subChunk)
	}

	return subChunks
}

// generateResponse generates the final response using LLM based on retrieved chunks
func (p *AgenticRAGProcessor) generateResponse(ctx context.Context, query string, chunks []DocumentChunk, options AgenticRAGOptions) (string, int, error) {
	if len(chunks) == 0 {
		return "I don't have enough information to answer your question.", 0, nil
	}

	// Build context from relevant chunks
	contextBuilder := strings.Builder{}
	contextBuilder.WriteString("Based on the following relevant information:\n\n")

	for i, chunk := range chunks {
		contextBuilder.WriteString(fmt.Sprintf("Source %d:\n%s\n\n", i+1, chunk.Content))
	}

	// Create a sophisticated prompt for response generation
	prompt := fmt.Sprintf(`You are an expert AI assistant that provides accurate, comprehensive answers based on provided context.

Context Information:
%s

User Question: %s

Instructions:
1. Answer the question using ONLY the information provided in the context
2. Be comprehensive but concise
3. If the context doesn't contain enough information to answer fully, state what you can answer and what information is missing
4. Cite which sources support your statements (e.g., "According to Source 1...")
5. If the question cannot be answered with the given context, clearly state this

Answer:`, contextBuilder.String(), query)

	// Generate response using LLM
	var response *ai.ModelResponse
	var err error

	if p.config.Model != nil {
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModel(p.config.Model),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     float64(options.Temperature),
				MaxOutputTokens: 2048,
			}),
		)
	} else {
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModelName(p.config.ModelName),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     float64(options.Temperature),
				MaxOutputTokens: 2048,
			}),
		)
	}

	if err != nil {
		return "", 0, fmt.Errorf("failed to generate response: %w", err)
	}

	// Count tokens used (approximation)
	tokensUsed := len(strings.Fields(prompt)) + len(strings.Fields(response.Text()))

	return response.Text(), tokensUsed, nil
}

// buildKnowledgeGraph extracts entities and relations from chunks using LLM
func (p *AgenticRAGProcessor) buildKnowledgeGraph(ctx context.Context, chunks []DocumentChunk) (*KnowledgeGraph, error) {
	if !p.config.KnowledgeGraph.Enabled || len(chunks) == 0 {
		return nil, nil
	}

	// Combine chunk contents for analysis
	var contentBuilder strings.Builder
	for i, chunk := range chunks {
		contentBuilder.WriteString(fmt.Sprintf("Document %d:\n%s\n\n", i+1, chunk.Content))
	}

	// Create prompt for knowledge extraction
	entityTypes := strings.Join(p.config.KnowledgeGraph.EntityTypes, ", ")
	relationTypes := strings.Join(p.config.KnowledgeGraph.RelationTypes, ", ")

	prompt := fmt.Sprintf(`You are an expert knowledge graph extractor. Extract entities and relationships from the provided text.

Text to analyze:
%s

Extract the following:

ENTITIES (with types: %s):
- Identify important entities and classify them
- Include confidence score (0.0-1.0)
- Only include entities with confidence > %.2f

RELATIONS (with types: %s):
- Identify relationships between extracted entities
- Include confidence score (0.0-1.0)
- Only include relations with confidence > %.2f

Respond with JSON in this exact format:
{
  "entities": [
    {"id": "entity_1", "name": "Entity Name", "type": "ENTITY_TYPE", "confidence": 0.95},
    {"id": "entity_2", "name": "Another Entity", "type": "ENTITY_TYPE", "confidence": 0.87}
  ],
  "relations": [
    {"id": "rel_1", "subject": "entity_1", "predicate": "RELATION_TYPE", "object": "entity_2", "confidence": 0.90}
  ]
}`,
		contentBuilder.String(), entityTypes, p.config.KnowledgeGraph.MinConfidenceThreshold,
		relationTypes, p.config.KnowledgeGraph.MinConfidenceThreshold)

	// Generate knowledge graph using LLM
	var response *ai.ModelResponse
	var err error

	if p.config.Model != nil {
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModel(p.config.Model),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     0.1, // Low temperature for consistent extraction
				MaxOutputTokens: 2048,
			}),
		)
	} else {
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModelName(p.config.ModelName),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     0.1, // Low temperature for consistent extraction
				MaxOutputTokens: 2048,
			}),
		)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to extract knowledge graph: %w", err)
	}

	// Parse the LLM response
	var kgResponse struct {
		Entities  []Entity   `json:"entities"`
		Relations []Relation `json:"relations"`
	}

	responseText := response.Text()
	if err := json.Unmarshal([]byte(responseText), &kgResponse); err != nil {
		// Return empty knowledge graph if parsing fails
		return &KnowledgeGraph{
			Entities:  []Entity{},
			Relations: []Relation{},
			Metadata: map[string]interface{}{
				"extraction_error": err.Error(),
				"raw_response":     responseText,
			},
		}, nil
	}

	// Filter by confidence threshold
	filteredEntities := make([]Entity, 0)
	for _, entity := range kgResponse.Entities {
		if entity.Confidence >= p.config.KnowledgeGraph.MinConfidenceThreshold {
			filteredEntities = append(filteredEntities, entity)
		}
	}

	filteredRelations := make([]Relation, 0)
	for _, relation := range kgResponse.Relations {
		if relation.Confidence >= p.config.KnowledgeGraph.MinConfidenceThreshold {
			filteredRelations = append(filteredRelations, relation)
		}
	}

	return &KnowledgeGraph{
		Entities:  filteredEntities,
		Relations: filteredRelations,
		Metadata: map[string]interface{}{
			"extraction_method": "llm_based",
			"entity_types":      p.config.KnowledgeGraph.EntityTypes,
			"relation_types":    p.config.KnowledgeGraph.RelationTypes,
			"min_confidence":    p.config.KnowledgeGraph.MinConfidenceThreshold,
			"created_at":        time.Now(),
		},
	}, nil
}

// isCommonWord checks if a word is a common word that shouldn't be an entity
func (p *AgenticRAGProcessor) isCommonWord(word string) bool {
	commonWords := map[string]bool{
		"The": true, "This": true, "That": true, "And": true, "But": true,
		"Or": true, "For": true, "With": true, "By": true, "At": true,
		"In": true, "On": true, "To": true, "From": true, "Of": true,
		"As": true, "Is": true, "Are": true, "Was": true, "Were": true,
		"Be": true, "Been": true, "Being": true, "Have": true, "Has": true,
		"Had": true, "Do": true, "Does": true, "Did": true, "Will": true,
		"Would": true, "Could": true, "Should": true, "May": true, "Might": true,
		"Can": true, "Must": true, "Shall": true, "Here": true, "There": true,
		"Where": true, "When": true, "What": true, "Who": true, "Why": true,
		"How": true, "All": true, "Any": true, "Some": true, "Many": true,
		"Few": true, "More": true, "Most": true, "Other": true, "Such": true,
		"No": true, "Not": true, "Only": true, "Own": true, "Same": true,
		"So": true, "Than": true, "Too": true, "Very": true, "Just": true,
		"Now": true, "Also": true, "Its": true, "My": true, "Your": true,
		"His": true, "Her": true, "Our": true, "Their": true,
	}
	return commonWords[word]
}

// verifyFacts performs fact verification on the generated response using LLM
func (p *AgenticRAGProcessor) verifyFacts(ctx context.Context, answer string, chunks []DocumentChunk) (*FactVerification, error) {
	if len(chunks) == 0 {
		return nil, nil
	}

	// Build source context for verification
	var contextBuilder strings.Builder
	contextBuilder.WriteString("Source documents:\n\n")
	for i, chunk := range chunks {
		contextBuilder.WriteString(fmt.Sprintf("Source %d:\n%s\n\n", i+1, chunk.Content))
	}

	// Create prompt for fact verification
	prompt := fmt.Sprintf(`You are an expert fact-checker. Verify the factual accuracy of the given answer against the provided source documents.

Source Context:
%s

Answer to verify:
%s

Task:
1. Break down the answer into individual factual claims
2. For each claim, verify it against the source documents
3. Assign status: "verified" (supported by sources), "refuted" (contradicted by sources), or "inconclusive" (not addressed in sources)
4. Provide confidence score (0.0-1.0)
5. List evidence from sources that support or refute each claim

Respond with JSON in this exact format:
{
  "claims": [
    {
      "text": "Specific claim text",
      "status": "verified|refuted|inconclusive", 
      "confidence": 0.95,
      "evidence": ["Source 1: Supporting text", "Source 2: Additional evidence"]
    }
  ],
  "overall": "verified|partially_verified|unverified"
}`, contextBuilder.String(), answer)

	// Generate fact verification using LLM
	var response *ai.ModelResponse
	var err error

	if p.config.Model != nil {
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModel(p.config.Model),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     0.1, // Low temperature for consistent verification
				MaxOutputTokens: 2048,
			}),
		)
	} else {
		response, err = genkit.Generate(ctx, p.config.Genkit,
			ai.WithModelName(p.config.ModelName),
			ai.WithPrompt(prompt),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature:     0.1, // Low temperature for consistent verification
				MaxOutputTokens: 2048,
			}),
		)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to verify facts: %w", err)
	}

	// Parse the LLM response
	var verificationResponse struct {
		Claims  []Claim `json:"claims"`
		Overall string  `json:"overall"`
	}

	responseText := response.Text()
	if err := json.Unmarshal([]byte(responseText), &verificationResponse); err != nil {
		// Return basic verification if parsing fails
		return &FactVerification{
			Claims: []Claim{
				{
					Text:       answer,
					Status:     "inconclusive",
					Confidence: 0.5,
					Evidence:   []string{"Fact verification parsing failed"},
				},
			},
			Overall: "unverified",
			Metadata: map[string]interface{}{
				"verification_error": err.Error(),
				"raw_response":       responseText,
			},
		}, nil
	}

	return &FactVerification{
		Claims:  verificationResponse.Claims,
		Overall: verificationResponse.Overall,
		Metadata: map[string]interface{}{
			"verification_method": "llm_based",
			"source_count":        len(chunks),
			"verified_at":         time.Now(),
		},
	}, nil
}
