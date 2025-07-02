package main

import (
	"context"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/genkit"

	"github.com/ZanzyTHEbar/genkithandler"
	"github.com/ZanzyTHEbar/genkithandler/internal/agentic"
)

func main() {
	// Initialize GenKit
	ctx := context.Background()
	g, err := genkit.Init(ctx)
	if err != nil {
		log.Fatalf("Failed to initialize GenKit: %v", err)
	}

	// Register the agentic RAG plugin with default configuration
	if err := genkithandler.InitializeAgenticRAGWithDefaults(g); err != nil {
		log.Fatalf("Failed to initialize agentic RAG plugin: %v", err)
	}

	fmt.Println("Agentic RAG plugin initialized successfully!")

	// Example usage: create a processor and process a query
	config := genkithandler.DefaultAgenticRAGConfig()
	processor := genkithandler.NewAgenticRAGProcessor(config)

	// Example agentic RAG request
	request := agentic.AgenticRAGRequest{
		Query: "What are the key components of artificial intelligence?",
		Documents: []string{
			"Artificial intelligence (AI) consists of several key components including machine learning, natural language processing, computer vision, robotics, and expert systems. Machine learning enables systems to learn from data without explicit programming. Natural language processing allows computers to understand and generate human language. Computer vision gives machines the ability to interpret visual information. Robotics combines AI with physical systems to create autonomous agents. Expert systems capture and utilize domain-specific knowledge to solve complex problems.",
			"The field of AI has evolved significantly since its inception. Early AI focused on symbolic reasoning and rule-based systems. Modern AI emphasizes data-driven approaches, particularly deep learning and neural networks. These approaches have revolutionized applications in image recognition, speech processing, and game playing. The integration of big data and powerful computing resources has accelerated AI development across industries.",
		},
		Options: agentic.AgenticRAGOptions{
			MaxChunks:              20,
			RecursiveDepth:         3,
			EnableKnowledgeGraph:   true,
			EnableFactVerification: true,
			Temperature:            0.7,
		},
	}

	// Process the request
	response, err := processor.Process(ctx, request)
	if err != nil {
		log.Fatalf("Failed to process agentic RAG request: %v", err)
	}

	// Display results
	fmt.Printf("\n=== Agentic RAG Response ===\n")
	fmt.Printf("Answer: %s\n\n", response.Answer)

	fmt.Printf("Processing Metadata:\n")
	fmt.Printf("- Processing Time: %v\n", response.ProcessingMetadata.ProcessingTime)
	fmt.Printf("- Chunks Processed: %d\n", response.ProcessingMetadata.ChunksProcessed)
	fmt.Printf("- Recursive Levels: %d\n", response.ProcessingMetadata.RecursiveLevels)
	fmt.Printf("- Model Calls: %d\n", response.ProcessingMetadata.ModelCalls)
	fmt.Printf("- Tokens Used: %d\n\n", response.ProcessingMetadata.TokensUsed)

	fmt.Printf("Relevant Chunks (%d):\n", len(response.RelevantChunks))
	for i, chunk := range response.RelevantChunks {
		fmt.Printf("  %d. %s (Score: %.3f)\n", i+1, chunk.Chunk.Content[:min(100, len(chunk.Chunk.Content))]+"...", chunk.Chunk.RelevanceScore)
	}

	if response.KnowledgeGraph != nil {
		fmt.Printf("\nKnowledge Graph:\n")
		fmt.Printf("- Entities: %d\n", len(response.KnowledgeGraph.Entities))
		fmt.Printf("- Relations: %d\n", len(response.KnowledgeGraph.Relations))

		fmt.Printf("\nTop Entities:\n")
		for i, entity := range response.KnowledgeGraph.Entities {
			if i >= 5 { // Show only first 5
				break
			}
			fmt.Printf("  - %s (%s) [%.2f]\n", entity.Name, entity.Type, entity.Confidence)
		}
	}

	if response.FactVerification != nil {
		fmt.Printf("\nFact Verification:\n")
		fmt.Printf("- Overall Status: %s\n", response.FactVerification.Overall)
		fmt.Printf("- Claims Verified: %d/%d\n",
			countVerifiedClaims(response.FactVerification.Claims),
			len(response.FactVerification.Claims))
	}

	fmt.Println("\n=== Example completed successfully! ===")
}

// Helper function to count verified claims
func countVerifiedClaims(claims []agentic.Claim) int {
	count := 0
	for _, claim := range claims {
		if claim.Status == "verified" {
			count++
		}
	}
	return count
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
