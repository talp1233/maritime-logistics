class Port:
    """
    מחלקה שמייצגת נמל.
    """
    def __init__(self, name, max_depth):
        self.name = name
        self.max_depth = max_depth  # עומק מים מקסימלי במטרים
    
    def can_accommodate(self, ship_draft):
        """
        בודק אם הנמל יכול להכיל ספינה עם עומק שוקע מסוים.
        """
        return ship_draft <= self.max_depth


class Ferry:
    """
    מחלקה שמייצגת מעבורת.
    """
    def __init__(self, name, capacity, current_port):
        self.name = name
        self.capacity = capacity
        self.current_port = current_port
        self.passengers = 0

    def load_passenger(self, num_passengers):
        if self.passengers + num_passengers <= self.capacity:
            self.passengers += num_passengers
            print(f"Loaded {num_passengers} passengers. Total: {self.passengers}/{self.capacity}")
            return True
        else:
            available = self.capacity - self.passengers
            print(f"Cannot load {num_passengers} passengers. Only {available} spots available.")
            return False

    def unload_passenger(self, num_passengers):
        if num_passengers <= self.passengers:
            self.passengers -= num_passengers
            print(f"Unloaded {num_passengers} passengers. Remaining: {self.passengers}")
            return True
        else:
            print(f"Cannot unload {num_passengers} passengers. Only {self.passengers} on board.")
            return False

    def sail_to(self, destination_port, distance_km, speed_kmh):
        if distance_km <= 0 or speed_kmh <= 0:
            print("Invalid distance or speed")
            return None
        time_hours = distance_km / speed_kmh
        old_port = self.current_port
        self.current_port = destination_port
        print(f"{self.name} sailed from {old_port} to {destination_port} in {time_hours:.2f} hours.")
        return time_hours


# יצירת נמלים
ports = {
    "Piraeus": Port("Piraeus", 15.0),
    "Santorini": Port("Santorini", 8.0),
    "Mykonos": Port("Mykonos", 6.0),
    "Crete": Port("Crete", 12.0)
}

# יצירת מעבורת
ferry = Ferry("Blue Star", 500, "Piraeus")

# סימולציה פשוטה
print("=== Ferry Simulation ===")
print(f"Ferry {ferry.name} is at {ferry.current_port}")

# נשאל את המשתמש לאן להפליג
destination_input = input("Enter destination port (Piraeus/Santorini/Mykonos/Crete) or 'quit': ")

# נמיר לאותיות קטנות כדי שיהיה case-insensitive
destination_lower = destination_input.lower()

# נחפש את הנמל - נמיר את כל המפתחות לאותיות קטנות ונשווה
found_port = None
for port_name, port_obj in ports.items():
    if port_name.lower() == destination_lower:
        found_port = port_name
        break

if found_port:
    distance = float(input(f"Enter distance from {ferry.current_port} to {found_port} (km): "))
    speed = float(input("Enter speed (km/h): "))
    ferry.sail_to(found_port, distance, speed)
else:
    print("Invalid port or quit")