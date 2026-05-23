import { useState } from "react";
import type { EventItem } from "../api";

type Props = {
    events: EventItem[];
    onSelectEvent: (time: number) => void;
};

function FightTimeline({ events, onSelectEvent }: Props) {
    const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

    if (events.length === 0) {
        return (
            <div className="timeline">
                <h2>Fight Timeline</h2>
                <p style={{ color: "#94a3b8" }}>Ударов не найдено.</p>
            </div>
        );
    }

    return (
        <div className="timeline">
            <h2>Fight Timeline</h2>

            {events.map((event, index) => {
                const isActive = selectedIndex === index;
                const fighterClass = event.fighter === "red"
                    ? "fighter-tag-red"
                    : "fighter-tag-blue";

                return (
                    <div
                        key={index}
                        className={isActive ? "event-card active-event" : "event-card"}
                        onClick={() => {
                            setSelectedIndex(index);
                            onSelectEvent(event.time);
                        }}
                    >
                        <div>
                            <strong>{event.time.toFixed(2)}s</strong>
                            <span className={`fighter-tag ${fighterClass}`}>
                                {event.fighter.toUpperCase()}
                            </span>
                        </div>

                        <div>
                            {event.hand} {event.punch_type}
                            <br />
                            <small>{event.target} · {event.effectiveness}</small>
                        </div>

                        <div>
                            conf: {(event.confidence * 100).toFixed(0)}%
                        </div>
                    </div>
                );
            })}
        </div>
    );
}

export default FightTimeline;
