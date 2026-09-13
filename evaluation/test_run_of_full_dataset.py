import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag_pipe_line import rag_run


# Dataset
dataset_path = Path(__file__).parent / "golden_dataset_50.jsonl"

# File to save completed results
output_path = Path(__file__).parent / "rag_results.json"


# Load dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]
    print(f"Loaded {len(dataset)} questions from {dataset_path}")


# Load previous results
if output_path.exists():
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)
        print(f"Loaded {len(results)} previous results from {output_path}")
else:
    results = []


# Questions already completed
completed = {item["question"] for item in results}


print(f"Total questions: {len(dataset)}")
print(f"Already completed: {len(completed)}")
print(f"Remaining: {len(dataset) - len(completed)}")


for item in dataset:

    question = item["question"]

    # Skip completed questions
    if question in completed:
        continue

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)

    try:

        # Run actual RAG + Gemini
        result = rag_run(question)

        record = {
            "question": question,
            "answer": result["answer"],
            "expected": item["golden_answer"],
            "sources": result["sources"],
            "model": result["model"],
            "contexts": result["contexts"]
        }

        # Add result
        results.append(record)

        # Save immediately
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                indent=2,
                ensure_ascii=False
            )

        print("\nANSWER:")
        print(result["answer"])

        print("\nEXPECTED:")
        print(item["golden_answer"])

        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)

        print(f"\nSaved: {len(results)}/{len(dataset)}")

    except Exception as e:

        print("\nERROR:")
        print(e)

        print("\nProgress saved.")
        print("Run this script again later to continue.")

        break


print("\n" + "=" * 60)
print("EVALUATION RUN FINISHED")
print(f"Completed: {len(results)}/{len(dataset)}")