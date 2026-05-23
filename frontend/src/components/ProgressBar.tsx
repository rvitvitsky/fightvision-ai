import { PIPELINE_STAGES } from "../api";

type Props = {
    stage: string;
    progress: number;
};

function ProgressBar({ stage, progress }: Props) {
    const currentIndex = Math.max(
        0,
        PIPELINE_STAGES.findIndex((s) => s.key === stage)
    );
    const currentStage = PIPELINE_STAGES[currentIndex] ?? PIPELINE_STAGES[0];
    const pct = Math.max(0, Math.min(100, Math.round(progress * 100)));

    return (
        <div className="progress-card">
            <div className="progress-header">
                <h2>Analyzing video...</h2>
                <span className="progress-percent">{pct}%</span>
            </div>

            <div className="progress-track">
                <div
                    className="progress-fill"
                    style={{ width: `${pct}%` }}
                />
            </div>

            <p className="progress-stage">
                {currentStage.label}
            </p>

            <ol className="progress-steps">
                {PIPELINE_STAGES.filter((s) => s.key !== "queued").map((s, i) => {
                    const idx = i + 1; // skip "queued"
                    let cls = "step";
                    if (idx < currentIndex) cls += " step-done";
                    else if (idx === currentIndex) cls += " step-active";
                    return (
                        <li key={s.key} className={cls}>
                            {s.label}
                        </li>
                    );
                })}
            </ol>
        </div>
    );
}

export default ProgressBar;
