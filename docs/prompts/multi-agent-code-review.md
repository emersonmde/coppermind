# Multi-Agent Comprehensive Code Review

Use this prompt to conduct a thorough architectural and code quality review using multiple parallel exploration agents. This approach provides deep analysis of each module while synthesizing findings into prioritized recommendations.

## When to Use

- After major refactoring or workspace restructuring
- Before releases to catch issues early
- Periodic codebase health checks
- When onboarding to understand code quality

## Usage

Copy and paste the following prompt to Claude Code:

---

**Prompt:**

Perform a thorough and critical code review as a Senior Software Engineer. Use the Task tool to launch the following exploration agents **in parallel** to analyze different parts of the codebase:

**Agent 1 - Core Library Review:**
```
Thoroughly explore the coppermind-core crate at crates/coppermind-core/src/. Read all files completely and analyze:
1. Code organization and module structure
2. Complex functions that could be simplified (>50 lines)
3. Code duplication across files
4. Trait design and abstraction opportunities
5. Error handling patterns (panic vs Result, error granularity)
6. Test coverage and quality
Return a detailed report with specific file:line references for all issues found.
```

**Agent 2 - Embedding Module Review:**
```
Thoroughly explore the embedding module at crates/coppermind/src/embedding/. Read all files completely and analyze:
1. Complex multi-step functions that could be refactored
2. Code duplication between chunking adapters
3. Trait abstraction opportunities
4. Error handling consistency
5. Model loading and caching patterns
6. Chunking strategy implementations
Return a detailed report with specific file:line references focusing on maintainability and potential abstractions.
```

**Agent 3 - Components Module Review:**
```
Thoroughly explore the components module at crates/coppermind/src/components/. Read all files completely and analyze:
1. Component organization and Dioxus patterns
2. Code duplication between components (formatting, utilities)
3. Complex multi-step logic that could be extracted to hooks
4. State management patterns and signal usage
5. Opportunities for shared utilities
Return a detailed report with specific file:line references.
```

**Agent 4 - Platform Abstraction Review:**
```
Thoroughly explore these directories:
- crates/coppermind/src/storage/ (platform-specific storage)
- crates/coppermind/src/platform/ (platform abstraction)
- crates/coppermind/src/processing/ (file processing)
- crates/coppermind/src/workers/ (web workers)
Read all files and analyze:
1. Code duplication between platform implementations
2. Trait abstraction opportunities
3. Complex async patterns that could be simplified
4. Error handling consistency
5. Underutilized existing abstractions (like ResultExt)
Return a detailed report with specific file:line references.
```

**Agent 5 - Crawler Module Review:**
```
Thoroughly explore the crawler module at crates/coppermind/src/crawler/. Read all files completely and analyze:
1. Code organization
2. Complex functions that could be refactored (especially crawl_with_progress)
3. Error handling patterns
4. Trait abstraction opportunities
5. Any duplication with other modules
Return a detailed report with specific file:line references.
```

After receiving all agent reports, synthesize findings into a **prioritized review document** with:

### Priority Levels

**HIGH PRIORITY** - Immediate action required:
- Safety issues (panics where Results expected)
- Correctness bugs
- Significant code duplication (>40 LOC repeated)
- Dead code / unused parameters

**MEDIUM PRIORITY** - Address when touching these areas:
- Complex functions (>100 lines)
- Maintainability concerns
- Inconsistent patterns
- Missing error context

**LOW PRIORITY** - Nice to have:
- Minor optimizations
- Code clarity improvements
- Additional abstractions (only if justified)

### For Each Finding Include:
1. **Files:** Specific file:line references
2. **Issue:** Description with code example
3. **Justification:** Why this matters (performance/readability/scalability/maintainability)
4. **Fix:** Recommended solution with code example

### Conclude With:
1. Summary metrics table (duplication LOC, complex functions count, etc.)
2. Trait abstraction opportunities table with justification (implement vs consider vs skip)
3. Recommended action plan organized into phases with time estimates

---

## Review Principles

Only recommend changes that are **clearly justified**:

| Change Type | Justification Required |
|-------------|----------------------|
| Extract duplicate code | >40 LOC duplicated, or >3 occurrences |
| Add trait abstraction | Multiple implementations exist OR clear extension point |
| Refactor function | >100 lines OR >3 nesting levels OR mixed concerns |
| Change error handling | Inconsistent with same module OR safety issue |

**Avoid:**
- Premature abstraction
- Over-engineering for hypothetical futures
- Changes that add complexity without clear benefit

## Output

Save the complete review to `./temp_code_review.md` for reference and potential follow-up.

## Example Output Format

```markdown
## Senior Engineer Code Review: Coppermind Workspace

Based on analysis of ~X LOC across Y crates, here are prioritized findings:

---

### HIGH PRIORITY

#### 1. [Issue Title]
**Files:**
- `path/to/file.rs:10-25`
- `path/to/other.rs:50-65`

**Issue:** [Description]
```rust
// Current code showing the problem
```

**Justification:** [Why this matters]

**Fix:**
```rust
// Recommended solution
```

---

### MEDIUM PRIORITY
[Same format]

### LOW PRIORITY
[Same format]

### Summary Metrics
| Category | Count | Impact |
|----------|-------|--------|
| Critical duplication | X | ~Y LOC savings |
| Complex functions | X | Readability |
| ... | ... | ... |

### Trait Abstraction Opportunities
| Abstraction | Location | Justification | Recommendation |
|-------------|----------|---------------|----------------|
| ... | ... | ... | Implement/Consider/Skip |

### Recommended Action Plan

**Phase 1 - Quick Wins (X hours)**
1. ...

**Phase 2 - Error Handling (X hours)**
1. ...
```
