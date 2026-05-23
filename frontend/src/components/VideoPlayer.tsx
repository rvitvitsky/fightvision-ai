type Props = {
    videoUrl: string;
};

function VideoPlayer({ videoUrl }: Props) {
    return (
        <div className="video-container">

            <h2>
                Fight Video
            </h2>

            <video
                controls
                width="100%"
                className="video-player"
            >
                <source
                    src={`http://127.0.0.1:8000${videoUrl}`}
                />

                Your browser does not support video.
            </video>

        </div>
    );
}

export default VideoPlayer;