#!/bin/bash

# Set permissions on directories
find . -type d -exec chmod 755 {} \;

# Set permissions on files
find . -type f -exec chmod 644 {} \;

# Convert all bmp files (including all subdirectories of current folder) to jpg
find . -iname '*.bmp' -execdir mogrify -format jpg {} \;

# Delete all bmp files
find . -iname '*.bmp' -execdir rm {} \;
