import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    ResponsiveContainer
} from "recharts";

type DNA = {
    aggression: number;
    defense: number;
    mobility: number;
    pressure: number;
};

type Props = {
    dna: {
        red: DNA;
    };
};

function FighterDNA({ dna }: Props) {

    const data = [
        {
            metric: "Aggression",
            value: dna.red.aggression
        },
        {
            metric: "Defense",
            value: dna.red.defense
        },
        {
            metric: "Mobility",
            value: dna.red.mobility
        },
        {
            metric: "Pressure",
            value: dna.red.pressure
        }
    ];

    return (
        <div className="dna-container">

            <h2>
                Fighter DNA
            </h2>

            <div
                style={{
                    width: "100%",
                    height: 400
                }}
            >
                <ResponsiveContainer>

                    <RadarChart data={data}>

                        <PolarGrid />

                        <PolarAngleAxis
                            dataKey="metric"
                            tick={{
                                fill: "#f8fafc",
                                fontSize: 14
                            }}
                        />

                        <Radar
                            dataKey="value"
                            stroke="#f97316"
                            fill="#f97316"
                            fillOpacity={0.6}
                        />

                    </RadarChart>

                </ResponsiveContainer>

            </div>

        </div>
    );
}

export default FighterDNA;