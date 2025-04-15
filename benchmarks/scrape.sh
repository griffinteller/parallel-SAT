#!/bin/bash

echo "Cleaning existing files/directories starting with uf or uuf..."
find . -maxdepth 1 -type d \( -name "uf*" -o -name "uuf*" \) -exec rm -rf {} \;
find . -maxdepth 1 -type f \( -name "uf*" -o -name "uuf*" \) ! -name "*.tar.gz" -exec rm -f {} \;

wget -qO benchm.html https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html

grep -Eo 'u+f[0-9]+-[0-9]+' benchm.html > urls.txt

echo "Found the following URLs/folder names:"
cat urls.txt

while IFS= read -r url; do
    if [[ "$url" != http* ]]; then
        url="https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/$url.tar.gz"
    fi
    echo "Downloading $url ..."
    wget "$url"
done < urls.txt

for file in *.tar.gz; do
    if [ -f "$file" ]; then
        target="${file%.tar.gz}"
        echo "Unpacking $file into directory $target ..."
        mkdir -p "$target"
        tar -xvzf "$file" -C "$target"
    fi
done

echo "Cleaning up..."
find . -maxdepth 1 -type f \( -name "uf*" -o -name "uuf*" \) ! -name "*.tar.gz" -exec rm -f {} \;
rm -f *.tar.gz
rm -rf ai
rm -f benchm.html
rm -f urls.txt