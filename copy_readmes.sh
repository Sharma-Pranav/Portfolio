#!/bin/bash

# Create the Hugo content directory for projects if it doesn't exist
mkdir -p content/projects

# Loop through each folder inside the projects directory
for d in projects/*/ ; do
    project_name=$(basename "$d")  # Extract project folder name
    readme_path="$d/README.md"  # Define README path
    output_file="content/projects/$project_name.md"

    # Check if README.md exists in the project folder
    if [ -f "$readme_path" ]; then
        # Extract the first line of the README to use as a summary (optional)
        summary=$(head -n 1 "$readme_path" | sed 's/# //')

        # Write the front matter to the new markdown file
        cat <<EOF > "$output_file"
---
title: "$project_name"
date: $(date +%Y-%m-%d)
type: "projects"
layout: "single"
summary: "$summary"
menu:
  main:
    parent: "projects"
    weight: 1
_build:
  list: always
  render: always
---
EOF

        # Append the original README content (excluding the first line if used as a summary)
        tail -n +2 "$readme_path" >> "$output_file"

        echo "Updated: $output_file"
    else
        echo "No README.md found in $project_name"
    fi
done