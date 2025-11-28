"""
Main entry point for the maritime simulation
"""

from maritime_sim.models.ferry import Ferry
from maritime_sim.services.simulation import create_default_ports, find_port_by_name


def main():
    """פונקציה ראשית שמריצה את הסימולציה"""
    # יצירת נמלים
    ports = create_default_ports()
    
    # יצירת מעבורת
    ferry = Ferry("Blue Star", 500, "Piraeus")
    
    # סימולציה פשוטה
    print("=== Ferry Simulation ===")
    print(f"Ferry {ferry.name} is at {ferry.current_port}")
    
    # נשאל את המשתמש לאן להפליג
    destination_input = input("Enter destination port (Piraeus/Santorini/Mykonos/Crete) or 'quit': ")
    
    if destination_input.lower() == 'quit':
        print("Goodbye!")
        return
    
    # נחפש את הנמל
    found_port = find_port_by_name(ports, destination_input)
    
    if found_port:
        distance = float(input(f"Enter distance from {ferry.current_port} to {found_port} (km): "))
        speed = float(input("Enter speed (km/h): "))
        ferry.sail_to(found_port, distance, speed)
    else:
        print("Invalid port or quit")


if __name__ == "__main__":
    main()

