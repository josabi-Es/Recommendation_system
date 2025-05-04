import re
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Batch
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the web scraping function
from data_processing.web_scraping import scrape_hacker_news
# Import the tokenization function
from data_processing.tokenization import tokenize_titles

import spacy

nlp = spacy.load("en_core_web_sm")


def main():
    """
    Orchestrates the entire recommendation process:
    1. Web scraping
    2. Tokenization
    3. Qdrant setup
    4. User simulation
    5. Recommendation generation
    """

    # 1. Web Scraping
    try:
        scrape_hacker_news()  # Execute web scraping
        print("Web scraping completed and data should be in data/DataWebScraping.csv")
    except Exception as e:
        print(f"Error during web scraping: {e}")
        return  # Exit if scraping fails

    # 2. Tokenization
    try:
        tokenize_titles()  # Execute tokenization
        print("Tokenization completed and data should be in data/DataTokeneizar.csv")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return  # Exit if tokenization fails

    # 3. Load data and prepare Qdrant
    try:
        df = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)
        df = df[["link", "category", "short_description"]]

        df_2 = pd.read_csv("data/DataTokeneizar.csv")  # Read tokenized data
        df_2 = df_2[["title", "points", "num_pages", "keywords"]]

        model = SentenceTransformer("sentence-transformers/average_word_embeddings_komninos")

        qdrant = QdrantClient(":memory:")  # In-memory Qdrant instance

        qdrant.create_collection(
            collection_name="recom",
            vectors_config=VectorParams(size=300, distance=Distance.COSINE)
        )

        # Insert documents into Qdrant in batches
        batch_size = 128
        for i in range(0, len(df_2), batch_size):
            i_end = min(i + batch_size, len(df_2))
            df_batch = df_2.iloc[i:i_end]
            texts = [" ".join(eval(kw)) if isinstance(kw, str) else "" for kw in df_batch["keywords"]]
            vectors = model.encode(texts).tolist()
            ids = list(range(i, i_end))
            metadatas = df_batch.to_dict(orient="records")
            qdrant.upsert(collection_name="recom", points=Batch(ids=ids, vectors=vectors, payloads=metadatas))

    except Exception as e:
        print(f"Error during Qdrant setup: {e}")
        return

    # 4. Simulate user
    try:
        user = df.loc[
            ((df["category"] == "ENTERTAINMENT") | (df["category"] == "NEWS"))
            & (df["short_description"].str.contains("movie", case=False))
        ]

        if user.empty:
            raise ValueError("No items match the filters to simulate user.")
        else:
            # 1. Extract and Clean Short Descriptions
            short_descriptions = user["short_description"].tolist()

            def clean_text(text):
                if isinstance(text, str):  # Check if it's a string
                    text = re.sub(r"[^a-zA-Z\s]", "", text, re.IGNORECASE)  # Remove non-alphanumeric
                    text = text.lower()  # Lowercase
                    text = " ".join(text.split())  # Remove extra spaces
                    return text
                else:
                    return ""  # Return empty string for non-string values

            cleaned_descriptions = [clean_text(desc) for desc in short_descriptions]

            # 2. Vectorize using TF-IDF
            vectorizer = TfidfVectorizer(stop_words="english", max_features=100)  # Adjust max_features
            tfidf_matrix = vectorizer.fit_transform(cleaned_descriptions)

            # 3. Get Feature Names (Words)
            feature_names = vectorizer.get_feature_names_out()

            # 4. Calculate Average TF-IDF Scores for Each Word
            import numpy as np

            tfidf_array = tfidf_matrix.toarray()
            average_tfidf_scores = np.mean(tfidf_array, axis=0)

            # 5. Get Top 4 Words
            top_indices = average_tfidf_scores.argsort()[-4:][::-1]  # Indices of top 4
            top_words = feature_names[top_indices]

            print("Top 4 representative words:")
            print(top_words)

        user_vector = model.encode(top_words)
        user_history = np.mean(user_vector, axis=0).tolist()
    except Exception as e:
        print(f"Error during user simulation: {e}")
        return

    # 5. Get Recommendations
    try:
        hits = qdrant.query_points(collection_name="recom", query=user_history, limit=5)

        # 6. Print Recommendations
        print("\nTop 5 recommended Hacker News articles:")
        output_file_path = "outputs/recommendations.txt"  # Corrected path
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("Top 5 recommended Hacker News articles:\n\n")
            for i, hit in enumerate(hits.points, 1):
                payload = hit.payload
                if payload and "title" in payload:
                    title = payload["title"]
                    points = payload.get("points", "N/A")
                    result_text = f"{i}. Title: {title}\n   Points: {points}\n"
                    print(result_text)
                    f.write(result_text)
                else:
                    result_text = f"{i}. No title found for this recommendation.\n"
                    print(result_text)
                    f.write(result_text)
        print(f"Recommendations written to {output_file_path}")  # Confirmation message
    except Exception as e:
        print(f"Error during recommendation generation: {e}")


if __name__ == "__main__":
    main()