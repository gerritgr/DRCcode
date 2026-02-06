import re
import csv

INPUT_FILE = "variable_names_map.txt"
OUTPUT_FILE = "variable_names_map.csv"

# Regex:
# - Variable name starts at column 0, uppercase letters/numbers
# - Followed by spacing
# - Then a label (until at least two spaces before numeric columns)
VAR_LINE_REGEX = re.compile(
    r"^([A-Z0-9]+)\s{2,}(.+?)\s{2,}\d+"
)

rows = []

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.rstrip()
        match = VAR_LINE_REGEX.match(line)
        if match:
            var_name = match.group(1)
            var_label = match.group(2).strip()
            rows.append((var_name, var_label))

# Write CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["variable_name", "description"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} variables to {OUTPUT_FILE}")
