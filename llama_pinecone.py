from flask import Flask, request, jsonify
import os
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from dotenv import load_dotenv
load_dotenv('.env')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
app = Flask(__name__)

pc = Pinecone(api_key=PINECONE_API_KEY)
# pc.create_index(
#     name="testing",
#     dimension=1536,
#     metric="dotproduct",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
# )
pinecone_index = pc.Index("testing")
documents = SimpleDirectoryReader("./data").load_data()



vector_store = PineconeVectorStore(pinecone_index=pinecone_index, namespace="testing")

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)


@app.route('/llama/pinecone', methods=['POST'])
def llama_pinecone():
    data = request.json
    query = data.get('query')
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
