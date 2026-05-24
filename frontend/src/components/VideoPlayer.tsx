import type { RefObject } from "react";

type Props = {
    videoUrl: string;
    videoRef: RefObject<HTMLVideoElement | null>;
};

function VideoPlayer({ videoUrl, videoRef }: Props) {
    return (
        <div className="video-container">
            <div className="video-header">
                <div>
                    <p className="timeline-kicker">Generated cut</p>
                    <h2>Highlights нарезка</h2>
                </div>

                {videoUrl && (
                    <a
                        href={videoUrl}
                        target="_blank"
                        rel="noreferrer"
                        download
                        className="download-button"
                    >
                        Скачать видео
                    </a>
                )}
            </div>

            {videoUrl ? (
                <video
                    ref={videoRef}
                    controls
                    preload="metadata"
                    playsInline
                    className="video-player"
                    key={videoUrl}
                >
                    <source src={videoUrl} type="video/mp4" />
                    Браузер не смог воспроизвести видео. Нажми кнопку скачать видео.
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
