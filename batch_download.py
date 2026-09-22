import yt_dlp
import os

def download_test_suite():
    videos = [
        {"name": "test_01_urban_crowd", "query": "ytsearch1:POV walking shinjuku crowd short"},
        {"name": "test_02_suburban_path", "query": "ytsearch1:suburban neighborhood walking POV 5 minutes"},
        {"name": "test_03_rainy_night", "query": "ytsearch1:rainy night city walking POV 5 minutes"},
        {"name": "test_04_chest_mount", "query": "ytsearch1:white cane POV walking blindness short"}
    ]
    
    print("Starting Batch Download of Test Suite...")
    
    for vid in videos:
        output_file = f"{vid['name']}.mp4"
        if os.path.exists(output_file):
            print(f"Skipping {output_file}, already exists.")
            continue
            
        print(f"\nSearching & Downloading: {vid['name']}...")
        
        # Options: max 720p, auto-merge video and audio
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f"{vid['name']}.%(ext)s",
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([vid['query']])
            print(f"Successfully saved {output_file}")
        except Exception as e:
            print(f"Error downloading {vid['name']}: {e}")

if __name__ == "__main__":
    download_test_suite()

