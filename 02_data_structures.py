ports = ["Piraeus", "Santorini", "Mykonos", "Crete", "Rhodes"]
port_depths = {
    "piraeus": 15.0,
    "santorini": 8.0,
    "mykonos": 6.0,
    "crete": 12.0,
    "rhodes": 10.0,
}
def find_suitable_ports(ship_draft, port_depths):
    suitable = []
    for port, depth in port_depths.items():
        if depth >= ship_draft:
            suitable.append(port)
    return suitable

ship_draft = float(input("Enter the ship draft: "))
suitable_ports = find_suitable_ports(ship_draft, port_depths)
print("Suitable ports:", suitable_ports)

