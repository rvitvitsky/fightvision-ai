import { useMemo, useState } from "react";
import type { EventItem } from "../api";

type Props = {
    events: EventItem[];
    onSelectEvent: (time: number) => void;
};

function formatTime(totalSeconds: number) {
    const safe = Math.max(0, totalSeconds || 0);
    const minutes = Math.floor(safe / 60);
    const seconds = safe % 60;
    return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function FightTimeline({ events, onSelectEvent }: Props) {
    const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
    const [isOpen, setIsOpen] = useState(true);

    const counts = useMemo(() => {
        return events.reduce(
            (acc, event) => {
                if (event.fighter === "red") acc.red += 1;
                if (event.fighter === "blue") acc.blue += 1;
                return acc;
            },
            { red: 0, blue: 0 }
        );
    }, [events]);

    return (
        <aside className={isOpen ? "timeline timeline-open" : "timeline timeline-closed"}>
            <div className="timeline-header">
                <div>
                    <p className="timeline-kicker">Interactive</p>
                    <h2>Fight Timeline</h2>
                </div>

                <button
                    type="button"
                    className="timeline-toggle"
                    onClick={() => setIsOpen((value) => !value)}
                    aria-expanded={isOpen}
                >
                    {isOpen ? "Свернуть" : "Открыть"}
                </button>
            </div>

            <div className="timeline-stats">
                <span>{events.length} ударов</span>
                <span className="timeline-stat-red">RED {counts.red}</span>
                <span className="timeline-stat-blue">BLUE {counts.blue}</span>
            </div>

            {!isOpen ? (
                <button
                    type="button"
                    className="timeline-open-button"
                    onClick={() => setIsOpen(true)}
                >
                    Открыть список ударов
                </button>
            ) : events.length === 0 ? (
                <div className="timeline-empty">
                    <p>Ударов не найдено.</p>
                    <span>Попробуй снизить порог или загрузить другой фрагмент боя.</span>
                </div>
            ) : (
                <div className="timeline-scroll" role="list">
                    {events.map((event, index) => {
                        const isActive = selectedIndex === index;
                        const fighterClass = event.fighter === "red"
                            ? "fighter-tag-red"
                            : "fighter-tag-blue";

                        return (
                            <button
                                key={`${event.frame}-${index}`}
                                type="button"
                                role="listitem"
                                className={isActive ? "event-card active-event" : "event-card"}
                                onClick={() => {
                                    setSelectedIndex(index);
                                    onSelectEvent(event.time);
                                }}
                            >
                                <div className="event-time-block">
                                    <strong>{formatTime(event.time)}</strong>
                                    <span className={`fighter-tag ${fighterClass}`}>
                                        {event.fighter.toUpperCase()}
                                    </span>
                                </div>

                                <div className="event-main-block">
                                    <strong>{event.hand} {event.punch_type}</strong>
                                    <small>{event.target} · {event.effectiveness}</small>
                                </div>

                                <div className="event-confidence">
                                    {(event.confidence * 100).toFixed(0)}%
                                    <small>conf</small>
                                </div>
                            </button>
                        );
                    })}
                </div>
            )}
        </aside>
    );
}

export default FightTimeline;
