// API client для FightVision gateway. Асинхронная схема: загрузка → polling → результат.

const API_BASE =
    (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000";

export type EventItem = {
    time: number;
    frame: number;
    fighter: "red" | "blue";
    hand: "left" | "right";
    punch_type: "jab" | "hook" | "cross" | "uppercut";
    target: "head" | "body";
    effectiveness: "landed" | "blocked" | "miss";
    confidence: number;
};

export type FighterStats = {
    punches: number;
    accuracy: number;
    landed: number;
    blocked: number;
    missed: number;
};

export type FighterDNAData = {
    aggression: number;
    defense: number;
    mobility: number;
    pressure: number;
};

export type AnalysisResult = {
    video_url: string;
    highlights_url: string;
    events: EventItem[];
    summary: { red: FighterStats; blue: FighterStats };
    dna: { red: FighterDNAData; blue: FighterDNAData };
    model_info?: Record<string, unknown>;
};

export type JobStatus = {
    job_id: string;
    status: "queued" | "processing" | "done" | "failed";
    stage: string;
    progress: number;
    error?: string | null;
};

// Стадии пайплайна в порядке выполнения — для UI прогресса
export const PIPELINE_STAGES = [
    { key: "queued", label: "В очереди" },
    { key: "pose_extraction", label: "Извлечение поз (YOLO)" },
    { key: "features", label: "Расчёт фич" },
    { key: "event_detection", label: "Поиск ударов" },
    { key: "visual_rerank", label: "Визуальная переоценка" },
    { key: "mlp_rerank", label: "MLP-переоценка" },
    { key: "attributes", label: "Атрибуты ударов" },
    { key: "postprocess", label: "Финальный отбор" },
    { key: "highlights_render", label: "Сборка нарезки" },
    { key: "finished", label: "Готово" },
] as const;


// --- API calls ---------------------------------------------------------------


export async function uploadVideo(file: File): Promise<string> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const text = await safeReadError(response);
        throw new Error(text || "Upload failed");
    }

    const data = await response.json();
    return data.job_id as string;
}


export async function getJobStatus(jobId: string): Promise<JobStatus> {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    if (!response.ok) {
        const text = await safeReadError(response);
        throw new Error(text || "Failed to get job status");
    }
    return (await response.json()) as JobStatus;
}


export async function getJobResult(jobId: string): Promise<AnalysisResult> {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}/result`);
    if (!response.ok) {
        const text = await safeReadError(response);
        throw new Error(text || "Failed to get job result");
    }
    return (await response.json()) as AnalysisResult;
}


export function buildMediaUrl(path: string): string {
    // path вида '/media/results/{job}/highlights.mp4'
    if (!path) return "";
    if (path.startsWith("http")) return path;
    return `${API_BASE}${path}`;
}


// --- internals ---------------------------------------------------------------


async function safeReadError(response: Response): Promise<string> {
    try {
        const data = await response.json();
        return typeof data?.detail === "string" ? data.detail : JSON.stringify(data);
    } catch {
        try {
            return await response.text();
        } catch {
            return `HTTP ${response.status}`;
        }
    }
}
