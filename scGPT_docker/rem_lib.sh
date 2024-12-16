#!/bin/bash

# Define the file names
txtA="requirements.txt"
txtB="unwanted_lib.txt"
tempFile="new_req.txt"

# Remove lines from txtA that are present in txtB
grep -v -F -x -f "$txtB" "$txtA" > "$tempFile"

# Replace txtA with the filtered content
mv "$tempFile" "$txtA"

echo "Lines from $txtB have been removed from $txtA."
