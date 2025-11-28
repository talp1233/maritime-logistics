"""
Simulation service - handles ferry simulation logic
"""


def find_port_by_name(ports_dict, port_name):
    """
    מוצא נמל לפי שם (case-insensitive).
    
    Args:
        ports_dict: מילון של נמלים (key = שם נמל, value = אובייקט Port)
        port_name: שם הנמל לחיפוש
    
    Returns:
        שם הנמל מהמילון אם נמצא, אחרת None
    """
    port_name_lower = port_name.lower()
    for port_key, port_obj in ports_dict.items():
        if port_key.lower() == port_name_lower:
            return port_key
    return None


def create_default_ports():
    """
    יוצר מילון של נמלים ברירת מחדל.
    
    Returns:
        מילון של נמלים
    """
    from ..models.port import Port
    
    return {
        "Piraeus": Port("Piraeus", 15.0),
        "Santorini": Port("Santorini", 8.0),
        "Mykonos": Port("Mykonos", 6.0),
        "Crete": Port("Crete", 12.0)
    }

