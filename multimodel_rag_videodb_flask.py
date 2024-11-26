from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from videodb import connect, SceneExtractionType, Segmenter
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer

app = Flask(__name__)

# Load environment variables
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
VIDEO_DB_API_KEY = os.getenv('VIDEO_DB_API_KEY')

# Initialize VideoDB connection and memory
conn = connect()
coll = conn.get_collection()
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

def process_video(video_url, query):
    """Process video and return query response"""
    # Upload and index video
    print("Uploading video...")
    video = conn.upload(url=video_url)
    
    # Index transcription
    print("Indexing transcription...")
    transcription_index_id = video.index_spoken_words()
    
    # Index scenes
    print("Extracting scenes...")
    scene_index_id = video.index_scenes(
        extraction_type=SceneExtractionType.time_based,
        extraction_config={"time": 2, "select_frames": ["first", "last"]},
        prompt="Describe the scene in detail",
    )
    
    # Get transcript nodes
    print("Processing transcript...")
    nodes_transcript_raw = video.get_transcript(
        segmenter=Segmenter.time, length=60
    )
    nodes_transcript = [
        TextNode(
            text=node["text"],
            metadata={key: value for key, value in node.items() if key != "text"},
        )
        for node in nodes_transcript_raw
    ]
    
    # Get scene nodes
    print("Processing scenes...")
    scenes = video.get_scene_index(scene_index_id)
    nodes_scenes = [
        TextNode(
            text=node["description"],
            metadata={
                key: value for key, value in node.items() if key != "description"
            },
        )
        for node in scenes
    ]
    
    # Create index and query
    print("Creating index and processing query...")
    index = VectorStoreIndex(nodes_scenes + nodes_transcript)
    query_engine = index.as_query_engine(memory=memory)
    response = query_engine.query(query)
    
    # Extract timestamps
    relevant_timestamps = [
        [node.metadata["start"], node.metadata["end"]] 
        for node in response.source_nodes
    ]
    
    # Generate video stream URL
    stream_url = None
    if relevant_timestamps:
        stream_url = video.generate_stream(merge_intervals(relevant_timestamps))
    
    return str(response), stream_url, relevant_timestamps

def merge_intervals(intervals):
    """Helper function to merge overlapping intervals"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for interval in intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)
    return merged

@app.route('/process', methods=['POST'])
def process():
    """Handle video processing and querying"""
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        query = data.get('query')
        
        if not video_url or not query:
            return jsonify({
                "error": "Both video_url and query are required"
            }), 400
        
        response, stream_url, timestamps = process_video(video_url, query)
        
        return jsonify({
            "response": response,
            "stream_url": stream_url,
            "timestamps": timestamps
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)