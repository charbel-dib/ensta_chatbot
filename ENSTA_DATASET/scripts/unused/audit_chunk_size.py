import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "03_chunks" / "chunks.jsonl"

def bucket(n, step):
    return (n // step) * step

def main():
    sizes = []
    with INP.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = obj.get("text","") or ""
            sizes.append(len(t))

    if not sizes:
        print("No chunks.")
        return

    sizes.sort()
    n = len(sizes)
    def pct(p):
        return sizes[int(p*(n-1))]

    print("Chunks:", n)
    print("min/max chars:", sizes[0], sizes[-1])
    print("p10/p25/p50/p75/p90 chars:", pct(0.10), pct(0.25), pct(0.50), pct(0.75), pct(0.90))

    # histogram
    hist = Counter(bucket(s, 250) for s in sizes)
    print("\nHistogram (bucket=250 chars) top:")
    for b in sorted(hist):
        if hist[b] >= max(10, n//200):  # affiche les buckets significatifs
            print(f"{b:5d}-{b+249:5d}: {hist[b]}")

    # estimate tokens ~ chars/4
    print("\nApprox tokens:")
    print("p50 ~", pct(0.50)//4, "tokens")
    print("p90 ~", pct(0.90)//4, "tokens")

if __name__ == "__main__":
    main()
