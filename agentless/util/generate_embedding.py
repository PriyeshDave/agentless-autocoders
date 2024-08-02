import os
from openai import OpenAI

# Load OpenAI API key from environment variable
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)  # get API key from platform.openai.com

MODEL = "text-embedding-3-small"

# Read text data from sample.txt
file_path = "sample_dataset.txt"
with open(file_path, "r") as file:
    text_data = file.read()

# Split the text into smaller chunks if necessary (OpenAI has token limits)
def chunk_text(text, max_tokens=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = len(word) + 1
        current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

chunks = chunk_text(text_data)

# Generate embeddings for each chunk
embeddings = []
for chunk in chunks:
    response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small"  # Specify the model for embeddings
    )
    embeds = [record.embedding for record in response.data]



# Optionally, you can save the embeddings to a file
import json
with open("embeddings.json", "w") as outfile:
    json.dump(embeds, outfile)


