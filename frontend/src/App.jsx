export default function App() {
  const slots = [1, 0, 1, 1, 0, 0, 1, 0]

  return (
    <div style={{ padding: "20px" }}>
      <h1>ParkWise RL Dashboard</h1>

      <button>Add Car</button>

      <h2>Parking Slots</h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 60px)",
          gap: "10px",
          marginTop: "20px"
        }}
      >
        {slots.map((slot, index) => (
          <div
            key={index}
            style={{
              width: "60px",
              height: "60px",
              background: slot ? "red" : "green"
            }}
          />
        ))}
      </div>

      <h2>Metrics</h2>

      <p>Reward: 120</p>
      <p>Occupancy: 75%</p>
      <p>Wait Time: 4s</p>
    </div>
  )
}