from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from videodb import connect, SceneExtractionType, Segmenter
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex


app = Flask(__name__)


# Load environment variables
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
VIDEO_DB_API_KEY = os.getenv('VIDEO_DB_API_KEY')
print(VIDEO_DB_API_KEY)
# Initialize shared data for scene index IDs
app.config['SCENE_INDEX_IDS'] = {}
# Initialize VideoDB connection and memory
def get_connection():
    conn = connect()
    return conn

@app.route('/videos', methods=['GET'])
def list_videos():
    """Get all videos from collection"""
    conn = get_connection()
    all_videos = conn.get_collection().get_videos()
    videos_list = [{
        "title": vid.name,
        "url": vid.stream_url,
        "length": round(float(vid.length))
    } for vid in all_videos]
    
    return jsonify({"videos": videos_list})

@app.route("/video/<id>", methods=["GET"])
def get_video(id):
    """
    Get a single video by id from default collection
    """
    conn = get_connection()
    all_videos = conn.get_collection().get_videos()

    vid = next(vid for vid in all_videos if vid.id == id)

    print("vid", vid)
    vid.get_transcript()
    transcript_text = vid.transcript_text

    response = {
        "video": {
            "id": vid.id,
            "title": vid.name,
            "url": vid.stream_url,
            "length": round(float(vid.length)),
            "transcript": transcript_text,
        }
    }
    return response

@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload a video URL and index it"""
    try:
        data = request.get_json()
        video_url = data.get('video_url')

        if not video_url:
            return jsonify({"error": "video_url is required"}), 400

        # Connect to VideoDB and upload video
        conn = get_connection()
        print(f"Uploading video: {video_url}")
        video = conn.upload(url=video_url)
        coll = conn.get_collection()
        video = coll.get_video(video.id)

        # Index transcription
        print(f"Indexing transcription for video: {video.name}")
        transcription_index_id = video.index_spoken_words()

        # Index scenes
        print(f"Indexing scenes for video: {video.name}")
        scene_index_id = video.index_scenes(
            extraction_type=SceneExtractionType.time_based,
            extraction_config={"time": 2, "select_frames": ["first", "last"]},
            prompt="Describe the scene in detail"
        )

        # Store Scene Index ID
        app.config['SCENE_INDEX_IDS'][video.name] = scene_index_id

        return jsonify({
            "success": True,
            "title": video.name,
            "scene_index_id": scene_index_id,
            "message": "Video uploaded and indexed successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_video():
    """Query an indexed video by title"""
    try:
        data = request.get_json()
        video_title = data.get('title')
        query = data.get('query')

        if not video_title or not query:
            return jsonify({"error": "Both title and query are required"}), 400

        # Connect to VideoDB
        conn = get_connection()
        all_videos = conn.get_collection().get_videos()

        # Find the video by title
        video = next((vid for vid in all_videos if vid.name == video_title), None)
        if not video:
            return jsonify({"error": f"Video with title '{video_title}' not found"}), 404

        # Get transcript nodes
        nodes_transcript_raw = video.get_transcript(segmenter=Segmenter.time, length=60)
        nodes_transcript = [
            TextNode(
                text=node.get("text", ""),
                metadata={key: value for key, value in node.items() if key != "text"}
            )
            for node in nodes_transcript_raw
        ]

        # Get scene nodes from the stored Scene Index ID
        scene_index_id = app.config['SCENE_INDEX_IDS'].get(video_title)
        if not scene_index_id:
            return jsonify({"error": f"No scene index found for '{video_title}'"}), 404

        scenes = video.get_scene_index(scene_index_id)
        nodes_scenes = [
            TextNode(
                text=node.get("description", ""),
                metadata={key: value for key, value in node.items() if key != "description"}
            )
            for node in scenes
        ]

        # Create index and query
        index = VectorStoreIndex(nodes_scenes + nodes_transcript)
        query_engine = index.as_query_engine()
        response = query_engine.query(query)

        # Extract relevant timestamps
        relevant_timestamps = [
            [node.metadata["start"], node.metadata["end"]]
            for node in response.source_nodes
            if "start" in node.metadata and "end" in node.metadata
        ]

        # Generate video stream URL for the result
        stream_url = None
        if relevant_timestamps:
            stream_url = video.generate_stream(merge_intervals(relevant_timestamps))

        return jsonify({
            "response": str(response),
            "stream_url": stream_url,
            "timestamps": relevant_timestamps
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

if __name__ == '__main__':
    app.run(debug=True)
