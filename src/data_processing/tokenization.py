import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nlp = spacy.load("en_core_web_sm")


def tokenize_titles(csv_file_path="../data/DataWebScraping.csv", output_csv_path="../data/DataTokeneizar.csv"):
    """
    Tokenizes titles from a CSV file, extracts keywords, and saves the result to a new CSV.

    Args:
        csv_file_path (str, optional): Path to the input CSV file.
                                     Defaults to "data/DataWebScraping.csv".
        output_csv_path (str, optional): Path to save the output CSV.
                                      Defaults to "data/DataTokeneizar.csv".

    Returns:
        pandas.DataFrame: A DataFrame with the original data and a new 'keywords' column.
    """

    df = pd.read_csv(csv_file_path, delimiter=",")
    titles = df["title"].tolist()

    def clean_text(text):
        text = re.sub(r"http\S+", "", text)
        return text.strip()

    cleaned_titles = [clean_text(t) for t in titles]

    def spacy_tokenizer(sentence):
        doc = nlp(sentence)
        return [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"]
        ]

    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, max_features=1000)
    X = vectorizer.fit_transform(cleaned_titles)
    feature_names = vectorizer.get_feature_names_out()

    keywords_per_title = []
    for idx, row in enumerate(X.toarray()):
        top_indices = row.argsort()[::-1]
        top_keywords = [feature_names[i] for i in top_indices if row[i] > 0][:4]
        keywords_per_title.append(top_keywords[:2] if len(top_keywords) < 2 else top_keywords[:4])

    df["keywords"] = keywords_per_title
    df.to_csv(output_csv_path, encoding="utf-8", index=False)
    print(f"Tokenization completed and data saved to {output_csv_path}")
    return df


if __name__ == "__main__":
    # This block is executed only if the script is run directly
    tokenized_data = tokenize_titles()
    print(tokenized_data[["title", "keywords"]])