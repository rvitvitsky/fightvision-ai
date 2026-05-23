import { useRef, useState } from "react";
import "./App.css";
import { uploadVideo } from "./api";
import FightTimeline from "./components/FightTimeline";
import FightSummary from "./components/FightSummary";
import FighterDNA from "./components/FighterDNA";
import VideoPlayer from "./components/VideoPlayer";

function App() {
    const [selectedFile, setSelectedFile] =
        useState<File | null>(null);

    const [loading, setLoading] =
        useState(false);

    const [errorMessage, setErrorMessage] =
        useState<string | null>(null);

    const [result, setResult] =
        useState<any>(null);

    const videoRef =
        useRef<HTMLVideoElement>(null);

    function jumpToMoment(time: number) {
        if (!videoRef.current) {
            return;
        }

        videoRef.current.currentTime = time;
        videoRef.current.play();
    }

    async function handleAnalyze() {
        if (!selectedFile) {
            setErrorMessage("Выбери видео перед анализом.");
            return;
        }

        try {
            setLoading(true);
            setErrorMessage(null);

            const data = await uploadVideo(selectedFile);

            setResult(data);
        } catch (error) {
            console.error(error);
            setErrorMessage("Не удалось проанализировать видео. Проверь, что backend запущен.");
        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="app">
            <section className="hero">
                <p className="eyebrow">
                    FightVision AI
                </p>

                <h1>
                    AI-анализ боксерского боя
                </h1>

                <p className="subtitle">
                    Загрузи видео боя, чтобы получить таймлайн ударов, статистику и Fight DNA.
                </p>

                <div className="upload-card">
                    <input
                        type="file"
                        accept="video/*"
                        disabled={loading}
                        onChange={(e) => {
                            const file = e.target.files?.[0];

                            if (file) {
                                setSelectedFile(file);
                                setErrorMessage(null);
                            }
                        }}
                    />

                    <button
                        type="button"
                        onClick={handleAnalyze}
                        disabled={loading}
                    >
                        {loading ? "Analyzing..." : "Analyze fight"}
                    </button>
                </div>

                {selectedFile && !result && !loading && (
                    <div className="empty-state">
                        <h2>
                            Видео выбрано
                        </h2>

                        <p>
                            {selectedFile.name}
                        </p>

                        <span>
                            Нажми Analyze fight, чтобы построить разбор боя.
                        </span>
                    </div>
                )}

                {loading && (
                    <div className="loading-card">
                        <div className="spinner" />

                        <h2>
                            Analyzing video...
                        </h2>

                        <p>
                            Extracting pose, detecting punches and building fight analytics.
                        </p>
                    </div>
                )}

                {errorMessage && (
                    <div className="error-card">
                        {errorMessage}
                    </div>
                )}

                {result && (
                    <div className="dashboard">
                        <div className="top-row">
                            <VideoPlayer
                                videoUrl={result.video_url}
                                videoRef={videoRef}
                            />

                            <FightTimeline
                                events={result.events}
                                onSelectEvent={jumpToMoment}
                            />
                        </div>

                        <div className="bottom-row">
                            <FightSummary
                                summary={result.summary}
                            />

                            <FighterDNA
                                dna={result.dna}
                            />
                        </div>
                    </div>
                )}
            </section>
        </main>
    );
}

export default App;