import yt_dlp
import os

def download_test_suite():
    videos = [
        {"name": "test_01_urban_crowd", "query": "ytsearch1:POV walking shinjuku crowd short"},
        {"name": "test_02_suburban_path", "query": "ytsearch1:suburban neighborhood walking POV 5 minutes"},
        {"name": "test_03_rainy_night", "query": "ytsearch1:rainy night city walking POV 5 minutes"},
        {"name": "test_04_chest_mount", "query": "ytsearch1:white cane POV walking blindness short"},
        {"name": "test_05_san_francisco_street", "query": "https://www.youtube.com/watch?v=NSgUUzIXT4E"},
        {"name": "test_06_residential_walk", "query": "https://www.youtube.com/watch?v=Zj_Vv40vtTo"},
        {"name": "test_07_argentina_street", "query": "https://www.youtube.com/watch?v=T21A9GZD3Mc"},
        {"name": "test_08_market", "query": "https://youtube.com/shorts/LWfU232N5Qw?si=4gvlRB3laXF7zA5i"},
        {"name": "test_09", "query": "https://youtube.com/shorts/Eex6l6jeIR4?si=qW3FZX4lTtosuWtK"},
        {"name": "test_10", "query": "https://youtube.com/shorts/55g9NJq7dxA?si=6up5b59Qzw1a3F7x"}
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Starting Batch Download of Test Suite...")
    
    for vid in videos:
        output_file = os.path.join(script_dir, f"{vid['name']}.mp4")
        if os.path.exists(output_file):
            print(f"Skipping {os.path.basename(output_file)}, already exists.")
            continue
            
        print(f"\nSearching & Downloading: {vid['name']}...")
        
        # Options: max 720p, auto-merge video and audio
        outtmpl_path = os.path.join(script_dir, f"{vid['name']}.%(ext)s")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': outtmpl_path,
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

