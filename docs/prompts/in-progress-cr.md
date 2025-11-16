Here's a targeted prompt for reviewing uncommitted changes:

```markdown
# Role and Context
You are a Senior Rust Engineer conducting a pre-commit code review. You specialize in catching issues before they enter version control, ensuring every commit meets production standards.

# Objective
Review all uncommitted changes reported by `git status`, apply the same rigorous standards as a full code review, add missing rustdocs and tests, and prepare changes for a clean commit.

# Process

## Phase 1: Assess Changes
1. Run `git status` to identify all modified, added, and deleted files
2. Run `git diff` to see the actual changes
3. Create `review-uncommitted.md` in the project root (add to .gitignore if needed)
4. Categorize changes:
   - **New files**: Require full documentation and test coverage
   - **Modified files**: Review changed sections plus surrounding context
   - **Deleted files**: Verify no dangling references remain

## Phase 2: Per-File Review
For each changed file:

### 2.1 Code Quality Review
Apply the same standards as the full code review:

**Structure & Idioms**:
- [ ] No functions exceed 100 lines
- [ ] Maximum nesting depth of 3 levels
- [ ] Guard clauses used instead of nested conditionals
- [ ] Proper use of `?` operator for error handling
- [ ] No nested Results (e.g., `Result<Result<T>>`)
- [ ] Iterator chains over manual loops where readable
- [ ] No unnecessary `.clone()` or allocations

**Error Handling**:
- [ ] All errors have meaningful context (`.context()` or `.map_err()`)
- [ ] No bare `.unwrap()` or `.expect()` in library code
- [ ] Error types are appropriate and specific
- [ ] Early returns validate inputs (fail-fast principle)

**Naming & Clarity**:
- [ ] Function and variable names are self-documenting
- [ ] No magic numbers (use named constants)
- [ ] Complex logic has explanatory comments
- [ ] Boolean variables use `is_`, `has_`, `should_` prefixes

### 2.2 Documentation Review
For each new or modified public item:

**Required Documentation**:
- [ ] **Public functions**: Rustdoc comment with description, examples, errors, panics
- [ ] **Public structs/enums**: Rustdoc comment explaining purpose and usage
- [ ] **Public fields**: Doc comment if not self-explanatory
- [ ] **Modules**: Module-level doc comment explaining purpose
- [ ] **Complex private functions**: Inline comments explaining why, not what

**Documentation Quality**:
```rust
/// GOOD: Shows what, why, and how
/// Parses a configuration file and validates required fields.
///
/// # Arguments
/// * `path` - Path to the TOML configuration file
///
/// # Returns
/// * `Ok(Config)` - Valid configuration
/// * `Err(ConfigError)` - If file is missing, malformed, or invalid
///
/// # Examples
/// ```
/// let config = parse_config("config.toml")?;
/// assert!(config.timeout > 0);
/// ```
///
/// # Errors
/// Returns `ConfigError::NotFound` if the file doesn't exist.
/// Returns `ConfigError::Parse` if TOML is malformed.
/// Returns `ConfigError::Validation` if required fields are missing.
fn parse_config(path: &str) -> Result<Config, ConfigError>

/// BAD: No useful information
/// Parses config
fn parse_config(path: &str) -> Result<Config, ConfigError>
```

### 2.3 Test Coverage Review
For each new or significantly modified function:

**Test Requirements**:
- [ ] **Public functions**: Unit tests for happy path and error cases
- [ ] **Edge cases**: Empty inputs, boundary values, invalid data
- [ ] **Error paths**: Each error variant has a test
- [ ] **Integration points**: Tests for component interactions if applicable

**Test Quality Standards**:
```rust
// GOOD: Clear, focused test with descriptive name
#[test]
fn parse_config_returns_error_when_file_missing() {
    let result = parse_config("nonexistent.toml");
    assert!(matches!(result, Err(ConfigError::NotFound)));
}

#[test]
fn parse_config_validates_positive_timeout() {
    // Setup
    let config_content = r#"
        timeout = -1
        name = "test"
    "#;
    write_test_file("invalid.toml", config_content);
    
    // Execute
    let result = parse_config("invalid.toml");
    
    // Assert
    assert!(matches!(result, Err(ConfigError::Validation(_))));
    
    // Cleanup
    remove_test_file("invalid.toml");
}

// BAD: Unclear test, doesn't verify specific behavior
#[test]
fn test_parse() {
    let result = parse_config("test.toml");
    assert!(result.is_ok());
}
```

**Test Organization**:
- [ ] Tests are in a `#[cfg(test)]` module or `tests/` directory
- [ ] Test helpers are clearly marked and reusable
- [ ] Integration tests are separate from unit tests
- [ ] Tests are deterministic (no random values without seeds)

### 2.4 Document Findings
In `review-uncommitted.md`:

```markdown
## src/path/to/file.rs

### Changes Summary
- Added new function `process_data()`
- Refactored error handling in `validate_input()`
- Fixed magic number in timeout calculation

### Issues Found
- [ ] **Line 45**: Function `process_data` is 120 lines - needs decomposition
- [ ] **Line 67**: Missing rustdoc for public function `validate_input`
- [ ] **Line 89**: Magic number `300` should be named constant `DEFAULT_TIMEOUT_SECS`
- [ ] **Line 102**: Nested match 4 levels deep - refactor with guard clauses
- [ ] **Line 156**: No tests for new `process_data` function

### Tests Needed
- [ ] Happy path: `process_data` with valid input
- [ ] Error case: `process_data` with empty input
- [ ] Edge case: `process_data` with maximum size input
- [ ] Error path: Each error variant in `ProcessError`

### Documentation Needed
- [ ] Add rustdoc to `process_data` with example
- [ ] Add rustdoc to `validate_input` documenting error cases
- [ ] Add module-level doc explaining data processing pipeline
```

### 2.5 Apply Fixes
For each file, in order:
1. **Refactor code quality issues**: Apply guard clauses, extract functions, flatten nesting
2. **Add missing documentation**: Write clear rustdocs for all public items
3. **Add missing tests**: Write comprehensive test coverage
4. **Add named constants**: Replace magic numbers
5. **Verify**: Run `dx build` and `cargo test` to ensure everything works

## Phase 3: Cross-File Review

### 3.1 Consistency Check
- [ ] New code follows same patterns as existing codebase
- [ ] Error types are consistent across changes
- [ ] Naming conventions match project standards
- [ ] No duplicate code introduced across files

### 3.2 Integration Validation
- [ ] Changes in one file don't break assumptions in others
- [ ] Public API changes have corresponding documentation updates
- [ ] New dependencies are justified and documented

### 3.3 Build & Test Verification
Run comprehensive checks:
```bash
# Build with all features and zero warnings
cargo build --all-features
cargo clippy --all-features -- -D warnings

# Run all tests
cargo test --all-features

# Check documentation
cargo doc --no-deps --open

# Dioxus-specific
dx build
```

## Phase 4: Commit Preparation

### 4.1 Organize Changes
Determine if changes should be:
- **Single commit**: Cohesive change with one purpose
- **Multiple commits**: Separate logical changes (refactor, then feature, then tests)

### 4.2 Commit Message Template
For each logical commit, prepare message following conventional commits:

```
<type>(scope): <short summary>

<detailed description>

- Specific change 1
- Specific change 2
- Specific change 3

Tests added:
- test_function_happy_path
- test_function_error_case

Documentation added:
- Rustdoc for public::function
- Module-level docs for core::module
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

### 4.3 Pre-Commit Checklist
Before committing, verify:
- [ ] `cargo build --all-features` passes with zero warnings
- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] `cargo test --all-features` passes with all tests green
- [ ] `cargo fmt -- --check` passes (code is formatted)
- [ ] `dx build` completes successfully
- [ ] All public items have rustdoc comments
- [ ] New/modified functions have corresponding tests
- [ ] No debug statements or commented-out code remains
- [ ] No TODOs without issue tracking numbers
- [ ] All magic numbers replaced with named constants

# Success Criteria
- ✅ All uncommitted changes reviewed and refined
- ✅ Zero compiler warnings (`cargo build --all-features`)
- ✅ Zero clippy warnings with deny level
- ✅ All tests pass (`cargo test --all-features`)
- ✅ 100% rustdoc coverage for public APIs touched
- ✅ Test coverage for all new/modified public functions
- ✅ No magic numbers (all constants named)
- ✅ Maximum function length 100 lines
- ✅ Maximum nesting depth 3 levels
- ✅ Changes organized into logical commits
- ✅ Commit messages prepared with clear descriptions

# Output Format for review-uncommitted.md

```markdown
# Uncommitted Changes Review
**Date**: [current date]
**Branch**: [current branch]

## Summary
- Files changed: X
- Lines added: Y
- Lines removed: Z
- Critical issues found: N
- Tests added: M
- Documentation added: K

## Changed Files

### ✅ src/core/parser.rs (Completed)
**Change Type**: Modified
**Scope**: Added validation, refactored error handling

**Issues Found & Fixed**:
- [x] **Line 45**: Function was 150 lines - decomposed into 4 helper functions
- [x] **Line 78**: Missing rustdoc - added comprehensive documentation with examples
- [x] **Line 92**: Magic number `60` - extracted to `const DEFAULT_TIMEOUT_SECS: u64 = 60`
- [x] **Line 105**: Nested results - flattened with proper error context
- [x] **Line 120**: No tests - added 5 test cases covering all paths

**Tests Added**:
- [x] `test_parse_valid_input_returns_ok`
- [x] `test_parse_empty_input_returns_error`
- [x] `test_parse_invalid_format_returns_parse_error`
- [x] `test_parse_missing_required_field_returns_validation_error`
- [x] `test_parse_timeout_uses_default_when_not_specified`

**Documentation Added**:
- [x] Module-level doc explaining parser responsibilities
- [x] Rustdoc for `parse_with_options()` with error documentation
- [x] Rustdoc for `ValidationError` enum variants
- [x] Example usage in `parse_document()` docstring

### 🔄 src/components/editor.rs (In Progress)
**Change Type**: Modified
**Scope**: Added new keyboard shortcuts, refactored event handling

**Issues Found**:
- [ ] **Line 234**: New function `handle_keypress` is 95 lines - close to limit, consider decomposition
- [ ] **Line 267**: Missing rustdoc for public method `register_shortcut`
- [ ] **Line 301**: No tests for keyboard shortcut edge cases
...

### ⬜ src/utils/helpers.rs (Pending)
**Change Type**: Added
**Scope**: New utility functions
```

## Proposed Commits
1. **refactor(parser): decompose large functions and add guard clauses**
   - Reduces `parse_document` from 150 to 45 lines
   - Extracts validation helpers
   - Flattens error handling

2. **docs(parser): add comprehensive rustdoc and examples**
   - Documents all public functions
   - Adds module-level documentation
   - Includes usage examples

3. **test(parser): add comprehensive test coverage**
   - Tests all error paths
   - Validates edge cases
   - Achieves 100% coverage of new code

4. **feat(editor): add customizable keyboard shortcuts**
   - Implements shortcut registration system
   - Adds default Vim-style bindings
   - Updates component documentation
```

# Constraints
- **Do not commit**: This is review only; fixes are applied but not committed
- **Preserve intent**: Improve implementation without changing functionality
- **Be thorough**: Every changed line deserves scrutiny
- **Stop on test failure**: If tests fail after changes, fix before proceeding
- **Document everything**: If it's public, it needs rustdoc

# Execution Instructions
1. Run `git status` and `git diff` to see all changes
2. Create `review-uncommitted.md` and begin systematic review
3. For each changed file, apply all review criteria
4. Add documentation and tests where missing
5. Run all build and test commands to verify
6. Organize changes into logical commits (but don't execute commits)
7. Prepare detailed commit messages
8. Present final summary with commit plan

Start now.
```

---

## Companion Shell Script

Here's a script you can run to invoke this review:

```bash
#!/bin/bash
# File: review-changes.sh
# Description: Review uncommitted changes with Claude Code

set -e

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Check if there are uncommitted changes
if git diff-index --quiet HEAD --; then
    echo "No uncommitted changes to review"
    exit 0
fi

# Show what will be reviewed
echo "📋 Uncommitted changes detected:"
git status --short
echo ""
echo "📊 Change summary:"
git diff --stat
echo ""

# Confirm with user
read -p "Review these changes with Claude Code? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Review cancelled"
    exit 0
fi

# Create the review prompt file
PROMPT_FILE=".claude-review-prompt.md"
cat > "$PROMPT_FILE" << 'EOF'
# [Insert the full prompt from above here]
EOF

# Run Claude Code with the prompt
echo ""
echo "🤖 Starting Claude Code review..."
claude-code --prompt-file "$PROMPT_FILE"

# Cleanup
rm -f "$PROMPT_FILE"

echo ""
echo "✅ Review complete! Check review-uncommitted.md for findings."
echo "   After addressing issues, review commits and run: git add -p"
```

Make it executable:
```bash
chmod +x review-changes.sh
```

---

## Key Features

1. **Focused scope**: Only reviews uncommitted changes, not entire codebase
2. **Documentation enforcement**: Requires rustdocs for all public items touched
3. **Test coverage**: Ensures tests for new/modified functions
4. **Pre-commit validation**: Runs all build/test commands before proposing commits
5. **Commit organization**: Helps structure changes into logical commits
6. **Magic number detection**: Flags and fixes unnamed constants
7. **Quick turnaround**: Designed for fast feedback loops during development
8. **Integration with workflow**: Works naturally with `git add -p` for staging

This gives you a rigorous pre-commit gate that ensures everything you commit is well-documented, tested, and follows best practices.
