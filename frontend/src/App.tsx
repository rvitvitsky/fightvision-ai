import { useState } from "react";
import "./App.css";
import { uploadVideo } from "./api";

function App() {
    const [selectedFile, setSelectedFile] =
        useState<File | null>(null);

    const [loading, setLoading] =
        useState(false);

    const [result, setResult] =
        useState<any>(null);

    async function handleAnalyze() {
        if (!selectedFile) {
            return;
        }

        try {
            setLoading(true);

            const data =
                await uploadVideo(selectedFile);

            setResult(data);
        } catch (error) {
            console.error(error);
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
                    Загрузи видео боя
                </p>

                <div className="upload-card">

                    <input
                        type="file"
                        accept="video/*"
                        onChange={(e) => {
                            const file =
                                e.target.files?.[0];

                            if (file) {
                                setSelectedFile(file);
                            }
                        }}
                    />

                    <button
                        type="button"
                        onClick={handleAnalyze}
                    >
                        {
                            loading
                                ? "Analyzing..."
                                : "Analyze fight"
                        }
                    </button>

                </div>

                {
                    result &&
                    (
                        <pre
                            style={{
                                textAlign: "left",
                                marginTop: 30,
                                overflowX: "auto"
                            }}
                        >
                            {
                                JSON.stringify(
                                    result,
                                    null,
                                    2
                                )
                            }
                        </pre>
                    )
                }

            </section>

        </main>
    );
}

export default App;