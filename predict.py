from rag_pipe_line import rag_run

while True:
    input_query = input("Enter your query (or 'exit' to quit): ").strip()
    if input_query.lower() == "exit":
        break

    result = rag_run(input_query)
    print("Answer:")
    print(result["answer"])

    for i, source in enumerate(result["sources"], start=1):
        print(f"Source [{i}]: {source}")
