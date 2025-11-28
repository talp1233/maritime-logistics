"""
Ferry model - represents a ferry ship
"""


class Ferry:
    """
    מחלקה שמייצגת מעבורת.
    """
    def __init__(self, name, capacity, current_port):
        """
        יוצר מעבורת חדשה.
        
        Args:
            name: שם המעבורת
            capacity: קיבולת מקסימלית של נוסעים
            current_port: הנמל הנוכחי שבו המעבורת נמצאת
        """
        self.name = name
        self.capacity = capacity
        self.current_port = current_port
        self.passengers = 0

    def load_passenger(self, num_passengers):
        """
        טוען נוסעים למעבורת.
        
        Args:
            num_passengers: מספר הנוסעים לטעינה
        
        Returns:
            True אם הטעינה הצליחה, אחרת False
        """
        if self.passengers + num_passengers <= self.capacity:
            self.passengers += num_passengers
            print(f"Loaded {num_passengers} passengers. Total: {self.passengers}/{self.capacity}")
            return True
        else:
            available = self.capacity - self.passengers
            print(f"Cannot load {num_passengers} passengers. Only {available} spots available.")
            return False

    def unload_passenger(self, num_passengers):
        """
        מפריק נוסעים מהמעבורת.
        
        Args:
            num_passengers: מספר הנוסעים לפריקה
        
        Returns:
            True אם הפריקה הצליחה, אחרת False
        """
        if num_passengers <= self.passengers:
            self.passengers -= num_passengers
            print(f"Unloaded {num_passengers} passengers. Remaining: {self.passengers}")
            return True
        else:
            print(f"Cannot unload {num_passengers} passengers. Only {self.passengers} on board.")
            return False

    def sail_to(self, destination_port, distance_km, speed_kmh):
        """
        מפליגה לנמל אחר.
        
        Args:
            destination_port: נמל היעד
            distance_km: מרחק בקילומטרים
            speed_kmh: מהירות בקילומטרים לשעה
        
        Returns:
            זמן ההפלגה בשעות, או None אם המרחק/מהירות לא תקינים
        """
        if distance_km <= 0 or speed_kmh <= 0:
            print("Invalid distance or speed")
            return None
        time_hours = distance_km / speed_kmh
        old_port = self.current_port
        self.current_port = destination_port
        print(f"{self.name} sailed from {old_port} to {destination_port} in {time_hours:.2f} hours.")
        return time_hours

