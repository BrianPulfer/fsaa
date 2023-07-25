# Update the CITATION.cff, README.md, and pyproject.toml files with the given version number.
# Also updates the CITATION.cff date-released field to the current date.
# Usage: ./update-version.sh <version>

# Update the CITATION.cff file.
sed -i "s/^version: .*/version: $1/" CITATION.cff
sed -i "s/date-released: .*/date-released: $(date +%Y-%m-%d)/" CITATION.cff

# Update the README.md file.
sed -i "s/version = {.*}/version = {$1}/" README.md

# Update the pyproject.toml file.
sed -i "s/version = \".*\"/version = \"$1\"/" pyproject.toml
