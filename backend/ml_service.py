def analyze_video(video_path: str) -> dict:
    return {
        "video_url": "/media/demo.mp4",
        "events": [
            {
                "time": 11.2,
                "frame": 336,
                "fighter": "red",
                "hand": "left",
                "punch_type": "jab",
                "target": "head",
                "confidence": 0.92
            },
            {
                "time": 14.5,
                "frame": 435,
                "fighter": "blue",
                "hand": "right",
                "punch_type": "cross",
                "target": "body",
                "confidence": 0.81
            }
        ],
        "summary": {
            "red": {
                "punches": 23,
                "accuracy": 0.61
            },
            "blue": {
                "punches": 18,
                "accuracy": 0.53
            }
        },
        "dna": {
            "red": {
                "aggression": 82,
                "defense": 47,
                "mobility": 64,
                "pressure": 73
            },
            "blue": {
                "aggression": 69,
                "defense": 58,
                "mobility": 71,
                "pressure": 62
            }
        }
    }
