type FighterStats = {
    punches: number;
    accuracy: number;
    landed: number;
    blocked: number;
    missed: number;
};

type Props = {
    summary: {
        red: FighterStats;
        blue: FighterStats;
    };
};

function FighterCard({
    name,
    cardClass,
    stats,
}: {
    name: string;
    cardClass: string;
    stats: FighterStats;
}) {
    return (
        <div className={`fighter-card ${cardClass}`}>
            <h3>{name}</h3>
            <p>
                Punches: <strong>{stats.punches}</strong>
            </p>
            <p>
                Accuracy: <strong>{(stats.accuracy * 100).toFixed(0)}%</strong>
            </p>
            <div className="effectiveness-row">
                <span>Landed: {stats.landed}</span>
                <span>Blocked: {stats.blocked}</span>
                <span>Missed: {stats.missed}</span>
            </div>
        </div>
    );
}

function FightSummary({ summary }: Props) {
    return (
        <div className="summary-container">
            <h2>Fight Summary</h2>
            <div className="summary-grid">
                <FighterCard
                    name="Red Fighter"
                    cardClass="red-card"
                    stats={summary.red}
                />
                <FighterCard
                    name="Blue Fighter"
                    cardClass="blue-card"
                    stats={summary.blue}
                />
            </div>
        </div>
    );
}

export default FightSummary;
