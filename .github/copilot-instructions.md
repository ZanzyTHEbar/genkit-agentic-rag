- make sure to update all documentation, todos, and reports as you complete each task
- make sure to prefix all placeholder logic with `// TODO: ` to indicate incomplete work
- make sure to run `go fmt ./...` to format all code
- make sure to run `go vet ./...` to check for any issues
- make sure to implement robust table-driven tests for all core functionality
- make sure to run `go test ./...` to ensure all tests pass

Be surgical in your approach, focusing on one task at a time. Ensure that each task is completed fully before moving on to the next. This will help maintain clarity and prevent confusion.

## Required Packages

- make sure to use the latest version of all dependencies
- make sure to use the following packages:
  - `github.com/ZanzyTHEbar/errbuilder-go` for custom error handling
  - `github.com/ZanzyTHEbar/assert-lib` for assertions
  - `github.com/spf13/viper` for configuration management
  - `github.com/google/uuid` for UUID generation
  - `github.com/stretchr/testify` for testing utilities

If `go mod tidy` is ran before the dependencies are used, it will remove the dependencies that are not used. Therefore, make sure to add AND USE _all_ required dependencies before running `go mod tidy`.

ALL PLACEHOLDER LOGIC MUST BE IMPLEMENTED IN REAL-TIME. DO NOT LEAVE ANY PLACEHOLDER LOGIC UNIMPLEMENTED. DO NOT LEAVE ANY TODOs UNADDRESSED. PRIORITISE COMPLETING ALL TASKS IN REAL-TIME BEFORE MOVING ON TO NEW TASKS.

Always take a forward looking attitude and outlook. Avoid implementing "backwards compatablity" logic.

Do note remove a file if corrupted, go through line by line and properly fix it.

# Agentic RAG Flow

OpenAI's Agentic RAG Flow is a framework that combines retrieval-augmented generation (RAG) with agentic capabilities. This allows for the generation of responses based on both pre-existing knowledge and real-time information retrieval.

## Key Components

1. **Load the Entire Document** into the context window.
   1. Logic to determine the selected models context window
   2. Appropriately and intelligently manage context window using advanced prompting techniques
2. **Chunk the Document** into 20 chunks that respect sentence boundaries.
3. **Prompt the model** for which chunks might contain relevant information.
4. **Drill down** into the selected relevant chunks.
5. **Recursively call this function** until we reach paragraph-level content.
6. **Generate a response** based on the retrieved information.
7. **Build Knowledge Graph** based on context
   1. If memory is enabled
8. **Verify the answer** for factual accuracy.

- Advanced Prompt Engineering
- Dynamic Prompt Templating and Chaining
- Generate Custom Prompts
- Send Prompt to LLM
- Concurrent Processing - Goroutines and Channels
- Custom Agent Module
- Inter-Agent Messaging - Message Bus / Actor Model
- Mixture of Agents - Ensemble and Iterative Refinement
- RAG Workflow Integration - Vector Retrieval and Context Augmentation
- LLM Final Response Generation
- Structured Output - JSON Schema Validation
- Observability and Metrics Logging
- Self-Optimization and Adaptive Tuning
- Robust Error Handling and Retry Logic
- Return Final Response to Application

Must use Hexagonal Arcitecture
Must use Functional Options Pattern
Must use Golang Best Practices
