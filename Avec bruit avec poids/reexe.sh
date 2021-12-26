while IFS= read -r LINE; do
    python3 main.py "$LINE"
done < reexe.txt