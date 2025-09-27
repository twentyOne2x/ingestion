#!/bin/bash

# Define an array of file paths
FILES=(
"src/__init__.py"
"src/ingest_v2/__init__.py"
"src/ingest_v2/configs/__init__.py"
"src/ingest_v2/configs/init.py"
"src/ingest_v2/configs/settings.py"
"src/ingest_v2/entities/__init__.py"
"src/ingest_v2/entities/init.py"
"src/ingest_v2/entities/extract.py"
"src/ingest_v2/pipelines/__init__.py"
"src/ingest_v2/pipelines/init.py"
"src/ingest_v2/pipelines/build_children.py"
"src/ingest_v2/pipelines/build_parents.py"
"src/ingest_v2/pipelines/run_all.py"
"src/ingest_v2/pipelines/upsert_pinecone.py"
"src/ingest_v2/router/__init__.py"
"src/ingest_v2/router/cache.py"
"src/ingest_v2/router/enrich_parent.py"
"src/ingest_v2/schemas/__init__.py"
"src/ingest_v2/schemas/init.py"
"src/ingest_v2/schemas/child.py"
"src/ingest_v2/schemas/json_schemas.py"
"src/ingest_v2/schemas/parent.py"
"src/ingest_v2/segmenter/__init__.py"
"src/ingest_v2/segmenter/init.py"
"src/ingest_v2/segmenter/segmenter.py"
"src/ingest_v2/sources/__init__.py"
"src/ingest_v2/sources/init.py"
"src/ingest_v2/sources/pumpfun.py"
"src/ingest_v2/sources/rss.py"
"src/ingest_v2/sources/youtube.py"
"src/ingest_v2/tests/__init__.py"
"src/ingest_v2/tests/init.py"
"src/ingest_v2/tests/test_entities.py"
"src/ingest_v2/tests/test_ids.py"
"src/ingest_v2/tests/test_segmenter.py"
"src/ingest_v2/tests/test_validators.py"
"src/ingest_v2/tests/fixtures/__init__.py"
"src/ingest_v2/tests/fixtures/init.py"
"src/ingest_v2/transcripts/__init__.py"
"src/ingest_v2/transcripts/init.py"
"src/ingest_v2/transcripts/assemblyai_json_ms_to_rawjson.py"
"src/ingest_v2/transcripts/normalize.py"
"src/ingest_v2/utils/__init__.py"
"src/ingest_v2/utils/init.py"
"src/ingest_v2/utils/backoff.py"
"src/ingest_v2/utils/batching.py"
"src/ingest_v2/utils/hashing.py"
"src/ingest_v2/utils/ids.py"
"src/ingest_v2/utils/logging.py"
"src/ingest_v2/utils/pinecone_client.py"
"src/ingest_v2/utils/timefmt.py"
"src/ingest_v2/validators/__init__.py"
"src/ingest_v2/validators/init.py"
"src/ingest_v2/validators/runtime.py"
)

remove_comments="${1:-true}"  # Default to true if no parameter is provided

# Directory where logs are stored
log_dir="logs"

# Find the most recent log file
log_file=$(ls -t "$log_dir"/*.txt 2>/dev/null | head -n 1)

# Variable to determine if logs were processed
logs_processed=false

if [ -n "$log_file" ]; then
  # Extract the first error log and everything that follows from the most recent log file
  context=$(awk '/\[ERROR\]/ {flag=1} flag' "$log_file")
  pre_error_context=$(awk '/\[ERROR\]/ {for (i=NR-5; i<NR; i++) print lines[i%4]; flag=1} {lines[NR%4]=$0} flag' "$log_file")

  # Check if any errors were found
  if [ -n "$context" ]; then
    logs_processed=true
  fi
fi

# Specify the directory path of interest for the tree command
tree_dir="."  # Update this to your path of interest

{
echo "\`\`\`"  # Start triple backticks
for file in "${FILES[@]}"; do
    echo "File: $file"
    echo "---------------------------------"
    cat "$file"
    echo ""
    echo "================================="
    echo ""
done

# Print directory structure of the specified path excluding certain paths
tree "$tree_dir" -I "node_modules|public|components|lib|assets|venv|utils|logs|fonts|cache" -L 4

# Conditionally add the pre-error context and error section if logs were processed
if [ "$logs_processed" = true ]; then
    echo ""
    echo "Pre-Error Context (5 lines before the first error) and First Error Context from $log_file:"
    echo "---------------------------------"
    echo "$pre_error_context"
    echo "$context"
    echo "================================="
fi

echo "\`\`\`"  # End triple backticks
if [ "$logs_processed" = true ]; then
    echo "Please fix"
else
    echo "Given the above: "
fi
} | xclip -selection clipboard

echo "Logs and script content have been copied to clipboard."
