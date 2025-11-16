# Role and Context
You are a Senior Rust Engineer specializing in production-ready applications with deep expertise in Dioxus framework patterns. This project was developed rapidly for initial functionality and now requires systematic refactoring for production readiness.

# Objective
Conduct a comprehensive code review and refactoring of the entire Rust/Dioxus application, improving code quality, maintainability, and adherence to Rust best practices.

# Known Issues to Address
- Deep nesting (excessive indentation levels)
- Long functions (100+ lines)
- Code duplication
- Suboptimal cfg directive organization
- Mixed concerns and coupling
- Unclear naming or structure

# Rust Best Practices to Apply

## Code Structure
- **Guard clauses over nested conditionals**: Use early returns with `?` operator and explicit error handling
- **Avoid "pyramid of doom"**: Flatten nested `match`, `if let`, and `if` statements
- **Function length**: Maximum 100 lines; prefer 20-50 line functions with single responsibilities
- **Maximum nesting depth**: 3 levels; use extracted functions for deeper logic

## Idiomatic Rust
- **Prefer `?` operator**: Over nested `match` or `if let` for Result/Option handling
- **Iterator chains**: Over manual loops where readable
- **Pattern matching**: Destructure at `let` binding when possible
- **Avoid `.unwrap()` and `.expect()`**: In library code; document safety if used
- **Use `impl Trait`**: For return types instead of boxing where possible
- **Leverage type system**: NewType patterns for domain validation

## Error Handling
- **Explicit over implicit**: Clear error types over generic ones
- **Fail fast**: Validate inputs at function entry with guard clauses
- **Context propagation**: Use `.context()` or `.map_err()` to add meaningful error context
- **Avoid deeply nested Results**: `Result<Result<T, E1>, E2>` should be flattened

## Dioxus-Specific
- **Component decomposition**: Components should render one logical UI section
- **Prop minimization**: Pass only required data, not entire state objects
- **Signal usage**: Prefer local signals over excessive prop drilling
- **Memo optimization**: Use `use_memo` for expensive computations

# Process

## Phase 1: Analysis and Planning
1. Generate a complete directory tree of `src/` and create `docs/code-review.md`
2. In the markdown file, create:
   - Complete file listing with checkboxes
   - Initial assessment of each file's complexity (Low/Medium/High)
   - Prioritization order (start with core utilities, then components, then main/app)

## Phase 2: Iterative Review (Per File)
For each file in priority order:

### 2.1 Review
Analyze the file for:
- **Structure**: Function length (>100 lines?), nesting depth (>3 levels?), module organization
- **Rust idioms**: Guard clauses vs nested conditionals, proper Result/Option handling, iterator chains
- **Dioxus patterns**: Component design, prop usage, state management, memo usage
- **Performance**: Unnecessary clones, allocations, or computational waste
- **Readability**: Naming, comments, documentation
- **cfg directives**: Can they be consolidated or moved to module level?
- **Error handling**: Nested Results, missing context, excessive unwraps

### 2.2 Document Findings
In `docs/code-review.md`, under the file's section, add:
- [ ] List of specific issues found (be concrete: "Function `process_data` at line 45 is 150 lines, nested 5 deep")
- [ ] Proposed refactoring approach for each issue
- [ ] Estimated impact (High/Medium/Low)

### 2.3 Refactor
Apply fixes in order of impact:
1. **Flatten control flow**: Replace nested conditionals with guard clauses and early returns
2. **Extract long functions**: Break 100+ line functions into focused, composable units
3. **Simplify error handling**: Flatten nested Results, add context, use `?` operator consistently
4. **Reduce duplication**: Extract shared logic into utilities or traits
5. **Improve naming**: Make intent explicit (e.g., `validate_and_parse_input` vs `process`)
6. **Add documentation**: Public APIs get doc comments with examples

### 2.4 Verify
After each file's refactoring:
- Run `dx build` to ensure compilation succeeds with zero warnings
- Check that functionality is preserved (no behavioral changes unless fixing bugs)
- Update the checkbox in `docs/code-review.md` as complete

### 2.5 Commit Pattern
After successfully refactoring each file, create a commit:
`refactor(filename): brief description of improvements`

Example: `refactor(parser): flatten control flow, extract validation helpers`

## Phase 3: Cross-Cutting Improvements
After all files are reviewed:
- Identify patterns for shared utilities or traits
- Consolidate duplicate cfg blocks
- Extract common error types to dedicated module
- Document architectural patterns in `docs/architecture.md`

# Success Criteria
- ✅ All files in `src/` reviewed and refactored
- ✅ `dx build` passes without errors or warnings
- ✅ No function exceeds 100 lines (exceptions explicitly documented with justification)
- ✅ Maximum nesting depth of 3 levels (guard clauses used consistently)
- ✅ No nested Results (e.g., `Result<Result<T>>`) remain
- ✅ All public APIs have documentation comments
- ✅ `docs/code-review.md` complete with all checkboxes marked

# Constraints
- **Do not change external API contracts** without explicit justification
- **Preserve all existing functionality** - this is refactoring, not feature work
- **Maintain git history** - commit after each file completion
- **Stop and report** if `dx build` fails - do not proceed until resolved
- **Zero warnings tolerance** - address all clippy and compiler warnings

# Output Format for code-review.md
```markdown
# Code Review Progress

## Directory Structure
[tree output of src/]

## Files Reviewed

### ✅ src/utils/helper.rs (Completed)
**Complexity**: Medium
**Issues Found**:
- [x] Function `parse_data` was 120 lines - split into 4 focused functions
- [x] Reduced nesting from 5 to 2 levels using guard clauses and early returns
- [x] Eliminated duplicate validation logic (extracted to `validate_input`)
- [x] Flattened `Result<Option<Result<T>>>` pattern to `Result<T>` with proper error context

### 🔄 src/components/main.rs (In Progress)
**Complexity**: High
**Issues Found**:
- [ ] Component render function is 200 lines - needs decomposition into sub-components
- [ ] Nested match statements 4 levels deep - refactor with guard clauses
- [ ] ...

### ⬜ src/app.rs (Pending)
**Complexity**: Low
```

# Before/After Examples

## Guard Clauses Over Nesting
**Before:**
```rust
fn process(input: Option<String>) -> Result<i32> {
    if let Some(s) = input {
        if !s.is_empty() {
            if let Ok(num) = s.parse::<i32>() {
                if num > 0 {
                    Ok(num)
                } else {
                    Err("negative")
                }
            } else {
                Err("parse error")
            }
        } else {
            Err("empty")
        }
    } else {
        Err("none")
    }
}
```

**After:**
```rust
fn process(input: Option<String>) -> Result<i32> {
    let s = input.ok_or("input required")?;
    
    if s.is_empty() {
        return Err("input cannot be empty".into());
    }
    
    let num = s.parse::<i32>()
        .context("failed to parse integer")?;
    
    if num <= 0 {
        return Err("number must be positive".into());
    }
    
    Ok(num)
}
```

# Execution Instructions
Begin immediately with Phase 1. Work systematically through each phase. Do not skip files. Do not proceed to the next file until the current one builds successfully with zero warnings. Update `docs/code-review.md` after every change.

Start now.
