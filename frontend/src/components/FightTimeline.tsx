import { useState } from "react";

type Event = {
    time: number;
    fighter: string;
    hand: string;
    punch_type: string;
    confidence: number;
};

type Props = {
    events: Event[];
    onSelectEvent: (time: number) => void;
};

function FightTimeline({
    events,
    onSelectEvent
}: Props) {

    const [selectedIndex, setSelectedIndex] =
        useState<number | null>(null);

    return (
        <div className="timeline">

            <h2>
                Fight Timeline
            </h2>

            {
                events.map((event, index) => (
                    <div
                        key={index}
                        className={
                            selectedIndex === index
                                ? "event-card active-event"
                                : "event-card"
                        }
                        onClick={() => {

                            setSelectedIndex(index);

                            onSelectEvent(
                                event.time
                            );
                        }}
                    >
                        <div>

                            <strong>
                                {event.time}s
                            </strong>

                            {" "}

                            {event.fighter}

                        </div>

                        <div>

                            {event.hand}

                            {" "}

                            {event.punch_type}

                        </div>

                        <div>

                            confidence:

                            {" "}

                            {(
                                event.confidence * 100
                            ).toFixed(0)
                            }

                            %

                        </div>

                    </div>
                ))
            }

        </div>
    );
}

export default FightTimeline;