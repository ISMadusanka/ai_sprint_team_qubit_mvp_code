import json
import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_content(data: Dict[str, Any]) -> str:
    """Format dictionary data into a string representation."""
    return ','.join(f"{k}: {v}" for k, v in data.items())


def set_json_data(json_data: str):
    # Parse the JSON string
    data = json.loads(json_data)

    # Create documents from the data
    documents = []
    for day, details in data.items():
        # Convert all details into a string format for the content
        content = format_content(details)
        documents.append(Document(page_content=content, metadata={"day": day, **details}))

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store in Pinecone
    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME")
    )

    return len(texts)


if __name__ == '__main__':
    print("Starting data import...")

    # Your JSON data here...
    api_response = '''
    {
  "day1": {
    "demand": 589,
    "date": "2023-01-01",
    "day_of_week": "Sunday",
    "is_special_day": "New Year",
    "weather": "sunny"
  },
  "day2": {
    "demand": 561,
    "date": "2023-01-02",
    "day_of_week": "Monday",
    "is_special_day": false,
    "weather": "cloudy"
  },
  "day3": {
    "demand": 640,
    "date": "2023-01-03",
    "day_of_week": "Tuesday",
    "is_special_day": false,
    "weather": "rainy"
  },
  "day4": {
    "demand": 656,
    "date": "2023-01-04",
    "day_of_week": "Wednesday",
    "is_special_day": false,
    "weather": "sunny"
  },
  "day5": {
    "demand": 727,
    "date": "2023-01-05",
    "day_of_week": "Thursday",
    "is_special_day": false,
    "weather": "partly cloudy"
  },
  "day6": {
    "demand": 697,
    "date": "2023-01-06",
    "day_of_week": "Friday",
    "is_special_day": false,
    "weather": "windy"
  },
  "day7": {
    "demand": 640,
    "date": "2023-01-07",
    "day_of_week": "Saturday",
    "is_special_day": false,
    "weather": "sunny"
  },
  "day8": {
    "demand": 599,
    "date": "2023-01-08",
    "day_of_week": "Sunday",
    "is_special_day": false,
    "weather": "cloudy"
  },
  "day9": {
    "demand": 568,
    "date": "2023-01-09",
    "day_of_week": "Monday",
    "is_special_day": false,
    "weather": "rainy"
  },
  "day10": {
    "demand": 577,
    "date": "2023-01-10",
    "day_of_week": "Tuesday",
    "is_special_day": false,
    "weather": "sunny"
  }
}
    '''

    num_chunks = set_json_data(api_response)
    print(f"Successfully stored {num_chunks} chunks in vector store")