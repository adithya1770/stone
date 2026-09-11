cd ~/Projects/stone
find runtime -type f -name "*.py" | sort | while read -r f; do
  echo "=================================================="
  echo "FILE: $f"
  echo "=================================================="
  cat "$f"
  echo
done