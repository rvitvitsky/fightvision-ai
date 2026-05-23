type FighterStats = {
    punches: number;
    accuracy: number;
};

type Props = {
    summary: {
        red: FighterStats;
        blue: FighterStats;
    };
};

function FightSummary({ summary }: Props) {
    return (
        <div className="summary-container">

            <h2>
                Fight Summary
            </h2>

            <div className="summary-grid">

                <div className="fighter-card red-card">

                    <h3>
                        Red Fighter
                    </h3>

                    <p>
                        Punches:
                        {" "}
                        {summary.red.punches}
                    </p>

                    <p>
                        Accuracy:
                        {" "}
                        {(summary.red.accuracy * 100).toFixed(0)}
                        %
                    </p>

                </div>

                <div className="fighter-card blue-card">

                    <h3>
                        Blue Fighter
                    </h3>

                    <p>
                        Punches:
                        {" "}
                        {summary.blue.punches}
                    </p>

                    <p>
                        Accuracy:
                        {" "}
                        {(summary.blue.accuracy * 100).toFixed(0)}
                        %
                    </p>

                </div>

            </div>

        </div>
    );
}

export default FightSummary;