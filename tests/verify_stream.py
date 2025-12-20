import requests
import json
import sys
import os

def test_stream_analysis(video_path="dummy_video.mp4"):
    url = "http://localhost:8000/api/v1/analyze-stream"

    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    print(f"Uploading {video_path} to {url}...")

    with open(video_path, 'rb') as f:
        files = {'file': f}
        # Use stream=True to consume the response as it arrives
        with requests.post(url, files=files, stream=True) as response:
            if response.status_code != 200:
                print(f"Failed: {response.status_code} {response.text}")
                return

            print("Response stream started:")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        data = json.loads(decoded_line)
                        status = data.get("status", "unknown")
                        progress = data.get("progress", 0)
                        message = data.get("message", "")
                        print(f"[{status.upper()}] {progress*100:.1f}% - {message}")

                        if status == "complete":
                            print("Analysis Result Received (Keys):", data.get("result", {}).keys())
                    except json.JSONDecodeError:
                        print("Received invalid JSON:", decoded_line)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_stream_analysis(sys.argv[1])
    else:
        test_stream_analysis()
