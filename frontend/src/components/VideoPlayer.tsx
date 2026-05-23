import type { RefObject } from "react";

type Props = {
    videoUrl: string;
    videoRef: RefObject<HTMLVideoElement | null>;
};

function VideoPlayer({ videoUrl, videoRef }: Props) {
    return (
        <div className="video-container">
            <h2>Highlights нарезка</h2>

            {videoUrl ? (
                <video
                    ref={videoRef}
                    controls
                    className="video-player"
                    key={videoUrl}
                >
                    <source src={videoUrl} type="video/mp4" />
                </video>
            ) : (
                <div className="empty-state">
                    <p>Видео не сгенерировано</p>
                    <span>Похоже, что в бою не нашлось ударов.</span>
                </div>
            )}
        </div>
    );
}

export default VideoPlayer;
