class Ferry:
    def __init__(self, name, capacity, current_port):
        self.name = name
        self.capacity = capacity
        self.current_port = current_port
        self.passengers = 0

    def load_passenger(self, num_passengers):
        if self.passengers + num_passengers <= self.capacity:
            self.passengers += num_passengers
            print(f"loaded {num_passengers} passengers. total: {self.passengers}/{self.capacity}")
            return True
        else:
            available = self.capacity - self.passengers
            print(f"cannot load {num_passengers} passengers. total: {available}spot available")
            return False

    def unload_passenger(self, num_passengers):
        if num_passengers <= self.passengers:
            self.passengers -= num_passengers
            print(f"unloaded {num_passengers} passengers. remaining: {self.passengers}")
            return True
        else:
            print(f"cannot unload {num_passengers} passengers. only {self.passengers} on board.")
            return False

    def sail_to(self, destination_port, distance_km, speed_kmh):
        if distance_km <= 0 or speed_kmh <= 0:
            print("invalid distance or speed")
            return None
        time_hours = distance_km / speed_kmh
        old_port = self.current_port
        self.current_port = destination_port
        print(f"{self.name} sailed from {old_port} to {destination_port} for {time_hours:.2f} hours.")
        return time_hours


# הקוד הזה צריך להיות מחוץ למחלקה (ברמה העליונה)
ferry1 = Ferry("blue star", 500, "Piraeus")
print(f"Ferry: {ferry1.name}, Capacity: {ferry1.capacity}, Current Port: {ferry1.current_port}")
ferry1.load_passenger(200)
ferry1.load_passenger(350)
ferry1.sail_to("Santorini", 150, 30)
ferry1.unload_passenger(100)