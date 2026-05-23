import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
    uploadVideo,
    getJobStatus,
    getJobResult,
    buildMediaUrl,
    type AnalysisResult,
    type JobStatus,
} from "./api";
import FightTimeline from "./components/FightTimeline";
import FightSummary from "./components/FightSummary";
import FighterDNA from "./components/FighterDNA";
import VideoPlayer from "./components/VideoPlayer";
import ProgressBar from "./components/ProgressBar";

const POLL_INTERVAL_MS = 1500;

function App() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [jobId, setJobId] = useState<string | null>(null);
    const [status, setStatus] = useState<JobStatus | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const videoRef = useRef<HTMLVideoElement>(null);
    const pollTimer = useRef<number | null>(null);

    // ---------- polling effect ---------------------------------------------

    useEffect(() => {
        if (!jobId) return;

        let cancelled = false;

        async function tick() {
            try {
                const st = await getJobStatus(jobId!);
                if (cancelled) return;
                setStatus(st);

                if (st.status === "done") {
                    const res = await getJobResult(jobId!);
                    if (cancelled) return;
                    setResult(res);
                    setJobId(null);
                    return;
                }
                if (st.status === "failed") {
                    setErrorMessage(st.error ?? "Обработка завершилась ошибкой.");
                    setJobId(null);
                    return;
                }

                pollTimer.current = window.setTimeout(tick, POLL_INTERVAL_MS);
            } catch (err) {
                if (cancelled) return;
                console.error(err);
                setErrorMessage(
                    "Связь с backend'ом прервалась. Проверь что gateway запущен."
                );
                setJobId(null);
            }
        }

        void tick();

        return () => {
            cancelled = true;
            if (pollTimer.current !== null) {
                clearTimeout(pollTimer.current);
                pollTimer.current = null;
            }
        };
    }, [jobId]);

    // ---------- handlers ----------------------------------------------------

    function jumpToMoment(time: number) {
        if (!videoRef.current) return;
        videoRef.current.currentTime = time;
        void videoRef.current.play();
    }

    async function handleAnalyze() {
        if (!selectedFile) {
            setErrorMessage("Выбери видео перед анализом.");
            return;
        }

        try {
            setErrorMessage(null);
            setResult(null);
            setStatus({
                job_id: "",
                status: "queued",
                stage: "queued",
                progress: 0,
            });
            const newJobId = await uploadVideo(selectedFile);
            setJobId(newJobId);
        } catch (err) {
            console.error(err);
            const message =
                err instanceof Error
                    ? err.message
                    : "Не удалось отправить видео.";
            setErrorMessage(message);
            setStatus(null);
        }
    }

    function handleReset() {
        setSelectedFile(null);
        setJobId(null);
        setStatus(null);
        setResult(null);
        setErrorMessage(null);
    }

    const isProcessing =
        status !== null &&
        (status.status === "queued" || status.status === "processing");

    return (
        <main className="app">
            <section className="hero">
                <p className="eyebrow">FightVision AI</p>

                <h1>AI-анализ боксерского боя</h1>

                <p className="subtitle">
                    Загрузи видео боя — на выходе нарезка ударов с разметкой, статистика и Fight DNA.
                </p>

                <div className="upload-card">
                    <input
                        type="file"
                        accept="video/*"
                        disabled={isProcessing}
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) {
                                setSelectedFile(file);
                                setErrorMessage(null);
                                setResult(null);
                            }
                        }}
                    />

                    {!result ? (
                        <button
                            type="button"
                            onClick={handleAnalyze}
                            disabled={isProcessing || !selectedFile}
                        >
                            {isProcessing ? "Analyzing..." : "Analyze fight"}
                        </button>
                    ) : (
                        <button type="button" onClick={handleReset}>
                            Новое видео
                        </button>
                    )}
                </div>

                {selectedFile && !result && !isProcessing && !errorMessage && (
                    <div className="empty-state">
                        <h2>Видео выбрано</h2>
                        <p>{selectedFile.name}</p>
                        <span>
                            Нажми Analyze fight, чтобы построить разбор боя.
                        </span>
                    </div>
                )}

                {isProcessing && status && (
                    <ProgressBar
                        stage={status.stage}
                        progress={status.progress}
                    />
                )}

                {errorMessage && (
                    <div className="error-card">{errorMessage}</div>
                )}

                {result && (
                    <div className="dashboard">
                        <div className="top-row">
                            <VideoPlayer
                                videoUrl={buildMediaUrl(
                                    result.video_url || result.highlights_url
                                )}
                                videoRef={videoRef}
                            />

                            <FightTimeline
                                events={result.events}
                                onSelectEvent={jumpToMoment}
                            />
                        </div>

                        <div className="bottom-row">
                            <FightSummary summary={result.summary} />
                            <FighterDNA dna={result.dna} />
                        </div>
                    </div>
                )}
            </section>
        </main>
    );
}

export default App;
