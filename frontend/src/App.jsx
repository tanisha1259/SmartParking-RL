import { useState } from "react"

export default function App() {
  const [slots, setSlots] = useState([
    { id: "S1", occupied: true },
    { id: "S2", occupied: false },
    { id: "S3", occupied: true },
    { id: "S4", occupied: false },
    { id: "S5", occupied: false },
    { id: "S6", occupied: true }
  ])

  const [cars, setCars] = useState(["CAR101"])

  const addCar = () => {
    const nextCar = `CAR${101 + cars.length}`

    setCars([...cars, nextCar])

    const updatedSlots = [...slots]

    const freeSlot = updatedSlots.find(
      slot => !slot.occupied
    )

    if (freeSlot) {
      freeSlot.occupied = true
      setSlots(updatedSlots)
    }
  }

  return (
    <div style={{ padding: "20px" }}>
      <h1>ParkWise RL Dashboard</h1>

      <button onClick={addCar}>
        Add Car
      </button>

      <h2>Parking Slots</h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 120px)",
          gap: "10px"
        }}
      >
        {slots.map(slot => (
          <div
            key={slot.id}
            style={{
              padding: "20px",
              background: slot.occupied
                ? "red"
                : "green"
            }}
          >
            {slot.id}
            <br />
            {slot.occupied
              ? "Occupied"
              : "Free"}
          </div>
        ))}
      </div>

      <h2>Incoming Cars</h2>

      {cars.map(car => (
        <p key={car}>{car}</p>
      ))}

      <h2>Metrics</h2>

      <p>
        Occupancy:
        {
          slots.filter(
            slot => slot.occupied
          ).length
        }
        /
        {slots.length}
      </p>
    </div>
  )
}