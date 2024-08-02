import os
from pinecone import ServerlessSpec
from pinecone import Pinecone
from openai import OpenAI
import json
import openai


# Initialize Pinecone
spec = ServerlessSpec(cloud="aws", region="us-east-1")
pc = Pinecone(api_key="...")
# Load OpenAI API key from environment variable
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)  #

# Define the index name and dimension
index_name = "semantic-search-index"
dimension = 1536  # Assuming using a model like 'text-embedding-ada-002' which outputs 768-dimensional vectors

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embed-3-small
        metric='cosine',
        spec=spec
    )

# Connect to the index
index = pc.Index(index_name)

# Load saved embeddings
with open("embeddings.json", "r") as file:
    saved_embeddings = json.load(file)

# Prepare items to insert
items_to_insert = [(f"doc_{i}", embedding) for i, embedding in enumerate(saved_embeddings)]

# Insert the embeddings into Pinecone
index.upsert(vectors=items_to_insert)


# Function to generate embedding for a given text
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"  # Specify the model for embeddings
    )
    embeds = [record.embedding for record in response.data]
    return embeds

# New query text
query_text = """sqlmigrate wraps it's outpout in BEGIN/COMMIT even if the database doesn't support transactional DDL
Description

	(last modified by Simon Charette)

The migration executor only adds the outer BEGIN/COMMIT ​if the migration is atomic and ​the schema editor can rollback DDL but the current sqlmigrate logic only takes migration.atomic into consideration.
The issue can be addressed by
Changing sqlmigrate ​assignment of self.output_transaction to consider connection.features.can_rollback_ddl as well.
Adding a test in tests/migrations/test_commands.py based on ​an existing test for non-atomic migrations that mocks connection.features.can_rollback_ddl to False instead of overdidding MIGRATION_MODULES to point to a non-atomic migration.
I marked the ticket as easy picking because I included the above guidelines but feel free to uncheck it if you deem it inappropriate."""

# Generate embedding for the query text
query_embedding = get_embedding(query_text)

# Query the index for similar embeddings
query_result = index.query(vector=query_embedding, top_k=3)  # Retrieve top 3 most similar embeddings

# Print the query results
print(query_result)
for match in query_result['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")


