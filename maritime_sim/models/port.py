"""
Port model - represents a port/harbor
"""


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
        
        Args:
            ship_draft: עומק השוקע של הספינה במטרים
        
        Returns:
            True אם הנמל יכול להכיל את הספינה, אחרת False
        """
        return ship_draft <= self.max_depth

