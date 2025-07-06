class MilvusSearchCLI:
    # TODO: Initialize the CLI
    def __init__(self, embedder, data_manager, df_metadata=None):
        """
        embedder: MilvusEmbedder instance - To convert text queries into vector embeddings
        data_manager: MilvusDataManager instance - To perform the actual vector search in Milvus
        df_metadata: Pandas DataFrame with metadata and 'id' column for filtering
                        - A DataFrame containing metadata corresponding to each vector
        """
        self.embedder = embedder
        self.data_manager = data_manager
        self.df_metadata = df_metadata

    # TODO: Run a semantic search for the given prompt, then filter results based on metadata if filters are provided.
    def search_with_filter(self, prompt, top_k=5, filters=None):
        query_vector = self.embedder.encode([prompt])[0]
        results = self.data_manager.search(query_vector, top_k=top_k)

        if filters and self.df_metadata is not None:
            filtered_results = []
            for res in results:
                row = self.df_metadata[self.df_metadata["id"] == res["id"]]
                if row.empty:
                    continue
                match = True
                for key, val in filters.items():
                    if (
                        key not in row.columns
                        or str(row.iloc[0][key]).lower() != str(val).lower()
                    ):
                        match = False
                        break
                if match:
                    filtered_results.append(res)
            results = filtered_results

        print(
            f"\nSearch results for: '{prompt}' (filtered by {filters})"
            if filters
            else f"\nSearch results for: '{prompt}'"
        )
        if not results:
            print("No matching results found.")
        else:
            for i, res in enumerate(results, 1):
                print(f"{i}. [Score: {res['score']:.4f}] {res['text']}")

    # TODO: Provide a user-friendly REPL (Read-Eval-Print Loop) CLI to perform searches and filtering until the user exits.
    def interactive_cli(self):
        print("\n--- Interactive Search CLI ---")
        print("Type 'exit' or 'quit' to stop.")
        print("You can add filters like: priority=High functionalarea=Login\n")

        while True:
            raw_input = input(
                "Enter your search prompt and optional filters:\n> "
            ).strip()
            if raw_input.lower() in ("exit", "quit"):
                break
            if not raw_input:
                continue

            parts = raw_input.split()
            prompt_parts = []
            filters = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    filters[k.lower()] = v
                else:
                    prompt_parts.append(part)
            prompt = " ".join(prompt_parts)
            if not prompt:
                print("Please enter a valid search prompt.")
                continue

            self.search_with_filter(prompt, top_k=5, filters=filters)

        print("Exiting CLI.")
