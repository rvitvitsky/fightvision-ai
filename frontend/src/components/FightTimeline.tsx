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
    return (
        <div className="timeline">

            <h2>
                Fight Timeline
            </h2>

            {
                events.map((event, index) => (
                    <div
                        key={index}
                        className="event-card"
                        onClick={() =>
                            onSelectEvent(
                                event.time
                            )
                        }
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
                            ).toFixed(0)}
                            %
                        </div>

                    </div>
                ))
            }

        </div>
    );
}

export default FightTimeline;