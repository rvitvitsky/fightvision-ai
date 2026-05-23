import {
    Radar,
    RadarChart,
    PolarGrid,
    PolarAngleAxis,
    ResponsiveContainer,
    Legend,
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
        blue: DNA;
    };
};

function FighterDNA({ dna }: Props) {
    const data = [
        {
            metric: "Aggression",
            red: dna.red.aggression,
            blue: dna.blue.aggression,
        },
        {
            metric: "Defense",
            red: dna.red.defense,
            blue: dna.blue.defense,
        },
        {
            metric: "Mobility",
            red: dna.red.mobility,
            blue: dna.blue.mobility,
        },
        {
            metric: "Pressure",
            red: dna.red.pressure,
            blue: dna.blue.pressure,
        },
    ];

    return (
        <div className="dna-container">
            <h2>Fighter DNA</h2>

            <div style={{ width: "100%", height: 400 }}>
                <ResponsiveContainer>
                    <RadarChart data={data}>
                        <PolarGrid stroke="rgba(148, 163, 184, 0.3)" />
                        <PolarAngleAxis
                            dataKey="metric"
                            tick={{ fill: "#f8fafc", fontSize: 14 }}
                        />
                        <Radar
                            name="Red"
                            dataKey="red"
                            stroke="#ef4444"
                            fill="#ef4444"
                            fillOpacity={0.45}
                        />
                        <Radar
                            name="Blue"
                            dataKey="blue"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.45}
                        />
                        <Legend
                            wrapperStyle={{ color: "#f8fafc", fontSize: 13 }}
                        />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}

export default FighterDNA;
