import type { RefObject } from "react";

type Props = {
    videoUrl: string;
    videoRef: RefObject<HTMLVideoElement | null>;
};

function VideoPlayer({
    videoUrl,
    videoRef
}: Props) {
    return (
        <div className="video-container">

            <h2>
                Fight Video
            </h2>

            <video
                ref={videoRef}
                controls
                className="video-player"
            >
                <source
                    src={`http://127.0.0.1:8000${videoUrl}`}
                />

            </video>

        </div>
    );
}

export default VideoPlayer;