# Coppermind UX & Design Specification

**Version:** 0.1
**Maintainer:** UX/Design
**Audience:** Software Engineers, Contributors, AI Agents

---

# 1. Overview

Coppermind is a **local-first semantic search engine** powered by Rust + WASM.
This document defines a complete **UX, visual identity, layout structure, and component system** for both:

* **Web Application** (browser)
* **Desktop Application** (future; same UI with platform-specific behaviors)

This spec is source-of-truth for all UI/UX decisions and must be used to:

* Implement the new UI
* Ensure consistency across features
* Guide future expansion (multi-index support, web crawling, browser plugin integrations)

---

# 2. Core Design Principles

### **2.1 Search-First**

Users primarily interact via search. The home screen must feel:

* Focused
* Calm
* Fast
* Technical yet approachable

As with Google, search dominates the UI.

### **2.2 Sleek Sci-Fi Terminal**

Primary aesthetic:

* Dark charcoal backgrounds
* Copper + neon-green accents
* Subtle glow treatments
* Clean, precise shapes and spacing
* Scifi-terminal but modern and elegant, not retro-kitschy

### **2.3 Technical Transparency**

Software engineers want to **see what the engine is doing**, but not be overwhelmed.

Defaults: simple
On demand: deep metrics and diagnostics

### **2.4 Non-Blocking Workflows**

Indexing should always be able to:

* Run in background
* Be paused/resumed later
* Be revisited through UI drawers or status indicators

### **2.5 Local-First Mental Model**

All processing happens on the device; UI should reinforce trust and privacy.

---

# 3. Information Architecture

Coppermind has two main user flows:

```
Search (Home)
  ↳ Metrics (panel/drawer)
  ↳ Show Result Details
  ↳ Show Source (platform-dependent)

Index
  ↳ Upload Files/Folders
  ↳ Show Indexing Progress
  ↳ Advanced Technical Details
  ↳ Previous Batch Summaries
```

Additional future flows the UI must anticipate:

* Multiple indexes (N-index)
* Web crawler indexing (URL input)
* Code/HTML specialized preprocessing
* Browser plugin integration (history/tabs)

---

# 4. Visual Language

## 4.1 Color System (CSS Variables)

```css
:root {
  /* Base */
  --cm-bg: #05070d;
  --cm-bg-elevated: #0b0f1a;
  --cm-bg-elevated-soft: #0f1523;

  /* Accents */
  --cm-accent-copper: #ff9a4a;
  --cm-accent-copper-soft: #f28b3f;
  --cm-accent-neon: #44ffb2;

  /* Text */
  --cm-text-primary: #f5f7ff;
  --cm-text-secondary: #a5b1d8;
  --cm-text-muted: #6b7699;

  /* Borders */
  --cm-border-subtle: rgba(255,255,255,0.05);
  --cm-border-strong: rgba(255,255,255,0.12);

  /* Status */
  --cm-status-ok-bg: rgba(68,255,178,0.1);
  --cm-status-ok-border: #44ffb2;
  --cm-status-warn-bg: rgba(255,154,74,0.1);
  --cm-status-warn-border: #ff9a4a;

  /* Metrics */
  --cm-metric-positive: #44ffb2;
  --cm-metric-neutral: #6b7699;
  --cm-metric-highlight: #ff9a4a;

  /* Glow */
  --cm-glow-soft: 0 0 18px rgba(68,255,178,0.18);
}
```

---

## 4.2 Typography

```css
:root {
  --cm-font-sans: system-ui, -apple-system, BlinkMacSystemFont,
                  "SF Pro Text", "Inter", sans-serif;

  --cm-font-size-xl: 2.2rem;
  --cm-font-size-lg: 1.4rem;
  --cm-font-size-md: 1rem;
  --cm-font-size-sm: 0.875rem;
  --cm-font-size-xs: 0.75rem;
}
```

Weights:

* Titles: medium (500–600)
* Body: regular (400)
* Metrics: semibold (600–700)

---

## 4.3 Shape & Motion

* Card radius: **16px**
* Buttons & pills: **999px** fully rounded
* Elevation:

    * Primary card shadow: `0 18px 40px rgba(0,0,0,0.5)`
    * Focus glow: copper → `color-mix(accent, transparent)`
* Animations: 150–200ms ease for hovers + transitions

---

# 5. Components

Each component is described at a level where the engineers or agents can reliably implement it.

---

## 5.1 App Bar (Global Header)

**Includes:**

* Coppermind wordmark (copper underline)
* Navigation:

    * Search (primary)
    * Index
* Metrics icon (opens metrics drawer)
* Optional: Index selector (future)

**Behavior:**

* Persistent across views
* Background: `--cm-bg-elevated`
* A subtle bottom border: `--cm-border-subtle`

---

## 5.2 Status Pills

Displayed below the header.

Examples:

* **Web Worker: ready**
* **Index: up to date**
* **Indexing in progress: 14 files**

Form:

```
● Label text
```

Colors driven by status tokens.

---

## 5.3 Search Card (Home)

Centered card containing:

* Index selector (future, disabled initially)
* Search bar (large)
* Submit button
* Supplemental hints:

    * # documents indexed
    * local-first message
    * hybrid semantic/keyword labels

Card background: `--cm-bg-elevated-soft`

---

## 5.4 Result Card

Reusable unit displaying:

* Rank: `#1`
* Title: clickable
* Fusion score (compact)
* Source meta (file name, index)
* Snippet preview (flexible length)
* Actions:

    * **Show Source**
    * **Details**

**Details expands inline**, showing:

* Vector score
* BM25 score
* RRF breakdown
* Chunk metadata
* Token count
* Path/URL (if available)

---

## 5.5 Metrics Panel (Developer Mode)

Toggled by metrics icon in the header.

### Content:

#### 1. Engine overview

* Total documents
* Total chunks
* Total tokens
* Avg tokens per chunk

#### 2. Live indexing metrics (if active)

* Tokens/sec
* Chunks/sec
* Avg time/chunk
* Playful text (e.g., “Vector engines spinning up…”)

### Behavior:

* Slides in below the app bar
* Does not obstruct search or results
* Dismissible via ESC or icon click

---

## 5.6 Indexing Components

### Upload Zone

* Large, inviting drop card
* Icon: folder + plus
* Shows:

    * Click to upload
    * Drag-and-drop
    * Multiple file/folder support
* Future: segmented control for `Files` | `URLs`

### File Row (Current Batch)

Shows:

* File name
* Phase text (“Embedding chunks… 4/6”)
* Progress bar (simple)
* Expand icon → advanced diagnostics

### Advanced Diagnostics (collapsed by default)

Includes:

* Chunk-by-chunk status
* Tokens per chunk
* Time per chunk/phase
* Any technical stats provided from engine

---

## 5.7 Show Source Panel

Unifies web and desktop behaviors.

### On Web:

* If file is remote URL → open new tab
* If file was uploaded → open Coppermind’s internal preview panel

    * Monospace text block
    * Metadata header
    * Scrollable
    * No editing

### On Desktop:

* If file is local path → open in Finder/Explorer using native APIs

Engineers should call the same component; implementation decides behavior.

---

# 6. Page Layouts

## 6.1 Search Home (Default)

```
[Top Bar]

[Status Pills]

[Search Card]

[Search Results]
  - Result Card #1
  - Result Card #2
  - ...
  - "Load more results" button

[Footer]
```

Empty state:
A central callout: *“No documents indexed yet. Start by indexing your files.”*
Button: → Index View

---

## 6.2 Index View

```
[Top Bar]

[Status Pills]

[Index Upload Zone]

[Current Batch]
   - File rows
   - Live progress

[Previous Batches]
   - Summary cards:
       CHUNKS: 6
       TOKENS: 2752
```

Background indexing supported:
A pill in header: **“Indexing 4 files… [View]”**

---

# 7. UX Behaviors (Critical Details)

## 7.1 Background Indexing

* Indexing continues whether on Search or Index views.
* Closing any modal/drawer does NOT stop indexing.
* UI should always reflect real indexing state from shared engine.

## 7.2 Infinite Scroll

* Results initially fetch fixed number (e.g. 10)
* “Load more” button (Google model)
* Design must allow future shift to auto-load on scroll

## 7.3 Multi-Index Architecture

UI placeholders for index selector:

* In search bar
* As global option in app bar

Even if disabled initially, layout must reserve space appropriately.

## 7.4 Tone & Microcopy

Coppermind uses **professional playful engineering tone**:

Examples:

* “Engines idle. Drop in some files to spin them up.”
* “Vectorizing chunks… ⚙️”
* “Fusion score computed with hybrid search strategy.”

Tone should be consistent across statuses, metrics, and index processes.

---

# 8. Future-Proofing Notes

Design must anticipate:

### Multi-index management

* Add/remove indexes
* Switch active indexes
* Per-index metrics

### URL-based indexing

* Additional input UI
* Crawl scope display

### Browser integrations

* History/tabs indexing
* Progressive indexing over time

### Code/HTML specialized chunking

* Different snippet types
* File-type specific iconography

---

# 9. Engineering Notes

These notes allow engineers/agents to interpret the spec correctly.

### 9.1 Web limitations (file paths)

Browsers do not expose full paths for uploaded files.
**Do not attempt** to reopen files via `file://`.

Solution:

* For web, display the **internal stored source text** in preview.
* Desktop builds may open Finder/Explorer.

### 9.2 Modal/drawer persistence

Indexing is engine-level, not component-level.
UI components must read from a shared `IndexingState`.

### 9.3 WASM Worker Constraints

UI should not block or depend on synchronous operations.

---

# 10. Implementation Guidance for Mockups

This spec is designed so **HTML/CSS/JS mockups can be generated from it** without loss of fidelity.

Each section maps to:

* Layout structure
* Component definitions
* Behavior rules
* Style tokens
* Visual language

Mockups should be implemented with:

* Semantic HTML
* Functional CSS using the variables above
* Minimal JS for interactions (expanders, drawers, toggles)

No framework assumptions required.

---

# 11. Deliverables From This Document

This spec enables generation of:

* Full web UI mockup (HTML/CSS/JS)
* Desktop UI version (same structure, different “Show Source” action)
* Component library (buttons, cards, metrics, expansions)
* Future feature designs (multi-index, URL indexing, plugins)

---

# 12. Versioning & Updates

This is **Version 0.1**.
Future updates should be appended as:

```
## Changelog
- v0.2: Multi-index UI pattern
- v0.3: URL indexing patterns
- v0.4: Desktop-specific adaptations
```
