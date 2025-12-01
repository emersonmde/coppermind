#!/bin/bash
# Generate a Coppermind evaluation dataset using Wikipedia articles.
#
# This script:
# 1. Fetches real Wikipedia articles across diverse topics (skips existing)
# 2. Uses Claude to generate varied query types for each article
#
# The dataset tests hybrid search with diverse query styles:
# - Short keyword queries (2-3 words)
# - Natural language questions
# - Conceptual/paraphrase queries (semantic understanding)
# - Synonym queries (vocabulary mismatch)
# - Typo queries (robustness)
# - Partial/incomplete queries (real user behavior)
#
# Prerequisites:
#   - ANTHROPIC_API_KEY environment variable set
#   - jq installed (brew install jq)
#   - curl installed
#
# Usage:
#   ./generate-dataset.sh [--output-dir DIR] [--regenerate-queries]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../data/coppermind-eval"
MODEL="claude-sonnet-4-20250514"
REGENERATE_QUERIES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --regenerate-queries)
            REGENERATE_QUERIES=true
            shift
            ;;
        --help)
            head -25 "$0" | tail -23
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY environment variable not set"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "Error: jq is required. Install with: brew install jq"
    exit 1
fi

echo "=== Generating Coppermind Evaluation Dataset ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# ~200 diverse Wikipedia articles across topics and lengths
ARTICLES=(
    # Science - Physics
    "Photosynthesis" "DNA" "Atom" "Electron" "Proton" "Neutron" "Light" "Sound"
    "Gravity" "Magnetism" "Electricity" "Energy" "Mass" "Velocity" "Acceleration"
    "Force" "Momentum" "Wave" "Frequency" "Wavelength"

    # Astronomy
    "Sun" "Moon" "Earth" "Mars" "Jupiter" "Saturn" "Venus" "Mercury_(planet)"
    "Neptune" "Uranus" "Pluto" "Asteroid" "Comet" "Galaxy" "Nebula" "Supernova"
    "Constellation" "Telescope"

    # Chemistry
    "Oxygen" "Hydrogen" "Carbon" "Nitrogen" "Iron" "Gold" "Silver" "Copper"
    "Aluminum" "Acid" "Base_(chemistry)" "Salt" "Water" "Chemical_reaction"
    "Molecule" "Ion" "Isotope"

    # Biology
    "Cell_(biology)" "Virus" "Bacteria" "Fungus" "Plant" "Animal" "Mammal" "Bird"
    "Fish" "Reptile" "Insect" "Spider" "Tree" "Flower" "Seed" "Protein" "Enzyme"
    "Chromosome" "Gene" "Mutation"

    # Human Body
    "Heart" "Brain" "Lung" "Liver" "Kidney" "Stomach" "Bone" "Muscle" "Blood"
    "Skin" "Eye" "Ear" "Nose" "Tooth" "Hair"

    # Geography
    "Ocean" "Mountain" "River" "Lake" "Desert" "Forest" "Island" "Volcano"
    "Earthquake" "Tsunami" "Hurricane" "Tornado" "Climate" "Weather" "Rain"
    "Snow" "Wind" "Cloud"

    # Countries
    "France" "Germany" "Italy" "Spain" "Japan" "China" "India" "Brazil"
    "Canada" "Australia" "Egypt" "Greece" "Russia" "Mexico"

    # History
    "Rome" "Athens" "Renaissance" "Pyramid" "Castle" "Knight" "Samurai" "Viking"
    "Gladiator" "Emperor" "King" "Queen" "War" "Revolution" "Democracy" "Republic" "Empire"

    # Technology
    "Computer" "Internet" "Software" "Hardware" "Algorithm" "Database" "Network"
    "Server" "Robot" "Satellite" "Radar" "Laser" "Semiconductor" "Transistor"
    "Microprocessor" "Smartphone" "Television" "Radio" "Camera"

    # Mathematics
    "Number" "Algebra" "Geometry" "Calculus" "Statistics" "Probability" "Equation"
    "Function_(mathematics)" "Triangle" "Circle" "Square" "Pi" "Infinity"
    "Prime_number" "Fraction" "Decimal" "Percentage"

    # Arts
    "Painting" "Sculpture" "Music" "Dance" "Theater" "Poetry" "Novel" "Film"
    "Photography" "Architecture" "Opera" "Ballet" "Jazz" "Symphony" "Guitar"
    "Piano" "Violin" "Drum"

    # Food
    "Bread" "Rice" "Wheat" "Corn" "Potato" "Tomato" "Apple" "Orange" "Banana"
    "Coffee" "Tea" "Wine" "Beer" "Chocolate" "Sugar" "Salt" "Pepper" "Cheese"
    "Milk" "Egg"

    # Sports
    "Football" "Basketball" "Baseball" "Tennis" "Golf" "Swimming" "Running"
    "Cycling" "Boxing" "Wrestling" "Gymnastics" "Skiing" "Skating" "Soccer"
    "Cricket" "Rugby"

    # Transportation
    "Car" "Airplane" "Ship" "Train" "Bicycle" "Motorcycle" "Bus" "Helicopter"
    "Submarine" "Rocket"
)

CORPUS_FILE="$OUTPUT_DIR/corpus.jsonl"
QUERIES_FILE="$OUTPUT_DIR/queries.jsonl"
QRELS_FILE="$OUTPUT_DIR/qrels.tsv"

# ============================================================================
# Step 1: Fetch Wikipedia articles (skip existing)
# ============================================================================
echo "Step 1/3: Fetching Wikipedia articles..."

touch "$CORPUS_FILE"

existing_docs=$(jq -r '._id' "$CORPUS_FILE" 2>/dev/null | sort -u)

fetch_wikipedia() {
    local title="$1"
    local max_retries=3
    local delay=1

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        local response=$(curl -s \
            "https://en.wikipedia.org/w/api.php?action=query&titles=${title}&prop=extracts&explaintext=1&format=json" 2>/dev/null)

        local extract=$(echo "$response" | jq -r '.query.pages | to_entries[0].value.extract // empty' 2>/dev/null)
        if [ -n "$extract" ] && [ "$extract" != "null" ]; then
            local page_title=$(echo "$response" | jq -r '.query.pages | to_entries[0].value.title // empty' 2>/dev/null)
            jq -c -n --arg title "$page_title" --arg extract "$extract" '{title: $title, extract: $extract}'
            return 0
        fi

        if [ $attempt -lt $max_retries ]; then
            sleep $delay
            delay=$((delay * 2))
        fi
    done

    echo ""
    return 1
}

doc_count=$(echo "$existing_docs" | grep -c . 2>/dev/null || echo "0")
new_docs=0
failed=0

for article in "${ARTICLES[@]}"; do
    doc_id=$(echo "$article" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -cd 'a-z0-9-')

    # Skip if already exists
    if echo "$existing_docs" | grep -q "^${doc_id}$"; then
        continue
    fi

    response=$(fetch_wikipedia "$article") || response=""
    extract=$(echo "$response" | jq -r '.extract // empty' 2>/dev/null) || extract=""
    display_title=$(echo "$response" | jq -r '.title // empty' 2>/dev/null) || display_title=""

    if [ -n "$extract" ] && [ "$extract" != "null" ] && [ -n "$display_title" ]; then
        doc_count=$((doc_count + 1))
        new_docs=$((new_docs + 1))

        jq -c -n \
            --arg id "$doc_id" \
            --arg title "$display_title" \
            --arg text "$extract" \
            '{_id: $id, title: $title, text: $text}' >> "$CORPUS_FILE"

        word_count=$(echo "$extract" | wc -w | tr -d ' ')
        echo "  [NEW] $display_title ($word_count words)"
    else
        failed=$((failed + 1))
        echo "  [SKIP] $article (fetch failed)"
    fi

    sleep 0.5
done

if [ $new_docs -eq 0 ]; then
    echo "  All articles already in corpus ($doc_count total)"
else
    echo "  Fetched $new_docs new articles ($failed failed, $doc_count total)"
fi

# ============================================================================
# Step 2: Generate queries using Claude
# ============================================================================
echo ""
echo "Step 2/3: Generating queries with Claude..."

# Check if we need to regenerate
if [ "$REGENERATE_QUERIES" = true ] || [ ! -f "$QUERIES_FILE" ] || [ ! -s "$QUERIES_FILE" ]; then
    echo "  Regenerating all queries..."
    > "$QUERIES_FILE"
    echo -e "query-id\tcorpus-id\tscore" > "$QRELS_FILE"
else
    echo "  Queries already exist. Use --regenerate-queries to regenerate."
    echo ""
    echo "Step 3/3: Validating dataset..."

    num_docs=$(wc -l < "$CORPUS_FILE" | tr -d ' ')
    num_queries=$(wc -l < "$QUERIES_FILE" | tr -d ' ')
    num_qrels=$(($(wc -l < "$QRELS_FILE" | tr -d ' ') - 1))

    echo ""
    echo "=== Dataset Already Complete ==="
    echo ""
    echo "Statistics:"
    echo "  Documents: $num_docs"
    echo "  Queries:   $num_queries"
    echo "  Qrels:     $num_qrels"
    echo ""
    echo "To regenerate queries with new types:"
    echo "  ./generate-dataset.sh --regenerate-queries"
    exit 0
fi

query_global_idx=0

generate_queries() {
    local doc_id="$1"
    local title="$2"
    local text="$3"

    local truncated_text="${text:0:1500}"

    local prompt="Given this Wikipedia article, generate search queries a user might use to find it.

Title: $title
Content: $truncated_text

Generate exactly 7 queries in these styles:
1. SHORT_KEYWORD: 2-3 word keyword query using key terms
2. QUESTION: Natural language question about the main topic
3. CONCEPTUAL: Describe the concept WITHOUT using the title word (test semantic understanding, e.g., for 'Photosynthesis' say 'process plants use to convert sunlight to energy')
4. SYNONYM: Use synonyms or related scientific/colloquial terms (not the exact title)
5. TYPO: A realistic misspelling of a key term (swap/missing letters, e.g., 'photosythesis' or 'chromosone')
6. PARTIAL: Incomplete query as if user stopped typing (e.g., 'photo synth' or 'dna doub')
7. QUESTION_DETAIL: Natural language question about a specific detail from the content

Return ONLY a valid JSON array:
[{\"type\":\"SHORT_KEYWORD\",\"text\":\"...\"},{\"type\":\"QUESTION\",\"text\":\"...\"},{\"type\":\"CONCEPTUAL\",\"text\":\"...\"},{\"type\":\"SYNONYM\",\"text\":\"...\"},{\"type\":\"TYPO\",\"text\":\"...\"},{\"type\":\"PARTIAL\",\"text\":\"...\"},{\"type\":\"QUESTION_DETAIL\",\"text\":\"...\"}]"

    local escaped_prompt=$(printf '%s' "$prompt" | jq -Rs .)

    local response=$(curl -s https://api.anthropic.com/v1/messages \
        -H "Content-Type: application/json" \
        -H "x-api-key: $ANTHROPIC_API_KEY" \
        -H "anthropic-version: 2023-06-01" \
        -d "{
            \"model\": \"$MODEL\",
            \"max_tokens\": 800,
            \"messages\": [{\"role\": \"user\", \"content\": $escaped_prompt}]
        }" 2>/dev/null)

    echo "$response" | jq -r '.content[0].text // empty' 2>/dev/null
}

doc_idx=0
doc_count=$(wc -l < "$CORPUS_FILE" | tr -d ' ')
queries_added=0

while IFS= read -r doc_json; do
    doc_id=$(echo "$doc_json" | jq -r '._id') || continue
    title=$(echo "$doc_json" | jq -r '.title') || continue
    text=$(echo "$doc_json" | jq -r '.text') || continue

    doc_idx=$((doc_idx + 1))
    echo "  [$doc_idx/$doc_count] Generating queries for: $title"

    queries_json=$(generate_queries "$doc_id" "$title" "$text") || queries_json=""

    if [ -n "$queries_json" ]; then
        clean_json=$(echo "$queries_json" | grep -o '\[.*\]' | head -1)

        if [ -n "$clean_json" ] && (echo "$clean_json" | jq -e '.' > /dev/null 2>&1); then
            while read -r query_obj; do
                query_type=$(echo "$query_obj" | jq -r '.type // "UNKNOWN"')
                query_text=$(echo "$query_obj" | jq -r '.text // empty')

                if [ -n "$query_text" ]; then
                    query_type_lower=$(echo "$query_type" | tr '[:upper:]' '[:lower:]')
                    query_id="q${query_global_idx}_${query_type_lower}"
                    query_global_idx=$((query_global_idx + 1))
                    queries_added=$((queries_added + 1))

                    jq -c -n --arg id "$query_id" --arg text "$query_text" \
                        '{_id: $id, text: $text}' >> "$QUERIES_FILE"
                    echo -e "$query_id\t$doc_id\t1" >> "$QRELS_FILE"
                fi
            done < <(echo "$clean_json" | jq -c '.[]' 2>/dev/null)
        else
            echo "    Warning: Failed to parse queries for $doc_id"
        fi
    else
        echo "    Warning: No response for $doc_id"
    fi

    sleep 0.2
done < "$CORPUS_FILE"

echo "  Generated $queries_added queries"

# ============================================================================
# Step 3: Summary
# ============================================================================
echo ""
echo "Step 3/3: Validating dataset..."

num_docs=$(wc -l < "$CORPUS_FILE" | tr -d ' ')
num_queries=$(wc -l < "$QUERIES_FILE" | tr -d ' ')
num_qrels=$(($(wc -l < "$QRELS_FILE" | tr -d ' ') - 1))

echo ""
echo "Query type distribution:"
for qtype in SHORT_KEYWORD QUESTION CONCEPTUAL SYNONYM TYPO PARTIAL QUESTION_DETAIL; do
    qtype_lower=$(echo "$qtype" | tr '[:upper:]' '[:lower:]')
    count=$(grep -c "_${qtype_lower}" "$QUERIES_FILE" 2>/dev/null || echo "0")
    printf "  %-18s %s\n" "$qtype_lower:" "$count"
done

echo ""
echo "=== Dataset Generation Complete ==="
echo ""
echo "Output files:"
echo "  $CORPUS_FILE"
echo "  $QUERIES_FILE"
echo "  $QRELS_FILE"
echo ""
echo "Statistics:"
echo "  Documents: $num_docs"
echo "  Queries:   $num_queries"
echo "  Qrels:     $num_qrels"
echo ""
echo "To run evaluation:"
echo "  rm -rf target/eval-cache && cargo run -p coppermind-eval --release"
