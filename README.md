# Recommendation System

This project provides a simple recommendation system that suggests Hacker News articles based on a simulated user history of preferences. It uses web scraping to gather data, natural language processing to analyze content, and a vector database to find similar articles.

## How It Works

This project provides a simple recommendation system that suggests Hacker News articles based on a simulated user history of preferences. The process involves the following steps:

1.  **Web Scraping for Data Collection:**
    
    * The application begins by scraping data directly from the Hacker News website (`https://hn.algolia.com`). This scraping process retrieves the titles and point counts of articles. This collected data forms the basis for the recommendations.

2.  **Title Tokenization and Keyword Extraction:**
    
    * Each article title obtained from web scraping is processed to extract its most important keywords.
    * Specifically, the titles are tokenized (split into words), and natural language processing techniques are applied to identify and retain the 2 to 4 most significant words. These keywords represent the core content of each article.

3.  **Simulated User History Creation:**
    
    * To simulate a user's reading history, the application utilizes a predefined dataset (`News_Category_Dataset_v3.json`).
    * This dataset contains a collection of news articles with categories and short descriptions.
    * Articles from this dataset that match certain criteria (e.g., belonging to "ENTERTAINMENT" or "NEWS" categories and containing the word "movie") are considered to represent the user's "past interests."

4.  **User Interest Vectorization:**
    
    * The short descriptions of the articles selected to represent the simulated user history are tokenized and transformed into numerical vectors.
    * This vectorization process converts the textual descriptions into a format that can be compared mathematically, allowing the application to understand the user's general interests.

5.  **Recommendation Generation via Vector Comparison:**
    
    * Finally, the vectors representing the user's simulated interests are compared to the vectors representing the Hacker News articles (derived from the extracted keywords).
    * This comparison is performed using a vector database (Qdrant) to identify the Hacker News articles that are most similar to the user's interests.
    * The top 5 most similar articles are then presented as recommendations.
## Prerequisites

Before running the application, make sure you have the following installed:

* Python 3.11
* pip (Python's package installer)

## Installation

1.  Clone the repository to your local machine.
2.  Navigate to the project directory in your terminal or command prompt.
3.  It's recommended to create a virtual environment:

    ```bash
    python -m venv .venv 
    ```

4.  Activate the virtual environment:

    * **Windows (PowerShell):**

        ```bash
        .\.venv\Scripts\activate
        ```

    * **Linux/macOS:**

        ```bash
        source .venv/bin/activate
        ```

5.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

6.  Download the required spaCy model:

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Usage

To run the application and generate recommendations, simply execute the main script:

```bash
python src/main.py
 ```

##  Roadmap

This project is under ongoing development. Here are some planned future improvements:

* **Automation:** Automate the recommendation workflow.
* **API:** Develop an API (e.g., with FastAPI).
* **UI:** Create a user interface for better display.
* **Enhanced Modeling:** Improve user preference modeling.
* **Scalability:** Optimize for handling more data.
* **Testing:**
    * Implement unit and integration tests using `pytest` to ensure code quality and prevent regressions.
    * Set up a CI/CD pipeline to automatically run tests on code changes.
