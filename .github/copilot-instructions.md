- make sure to update all documentation, todos, and reports as you complete each task
- make sure to prefix all placeholder logic with `// TODO: ` to indicate incomplete work
- make sure to run `go mod tidy` after adding new dependencies
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