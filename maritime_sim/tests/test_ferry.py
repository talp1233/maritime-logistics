"""
Unit tests for Ferry class
"""

import sys
import os

# הוספת הנתיב של הפרויקט כדי שנוכל לייבא את המודולים
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from maritime_sim.models.ferry import Ferry


def test_ferry_creation():
    """בודק שיצירת מעבורת עובדת"""
    ferry = Ferry("Test Ferry", 100, "Piraeus")
    assert ferry.name == "Test Ferry"
    assert ferry.capacity == 100
    assert ferry.current_port == "Piraeus"
    assert ferry.passengers == 0


def test_load_passengers():
    """בודק טעינת נוסעים"""
    ferry = Ferry("Test Ferry", 100, "Piraeus")
    
    # טעינה תקינה
    result = ferry.load_passenger(50)
    assert result == True
    assert ferry.passengers == 50
    
    # טעינה נוספת
    result = ferry.load_passenger(30)
    assert result == True
    assert ferry.passengers == 80
    
    # טעינה שתעבור את הקיבולת
    result = ferry.load_passenger(30)  # 80 + 30 = 110 > 100
    assert result == False
    assert ferry.passengers == 80  # לא השתנה


def test_unload_passengers():
    """בודק פריקת נוסעים"""
    ferry = Ferry("Test Ferry", 100, "Piraeus")
    ferry.load_passenger(50)
    
    # פריקה תקינה
    result = ferry.unload_passenger(30)
    assert result == True
    assert ferry.passengers == 20
    
    # פריקה שתעבור את מספר הנוסעים
    result = ferry.unload_passenger(30)  # רק 20 נוסעים
    assert result == False
    assert ferry.passengers == 20  # לא השתנה


def test_sail_to():
    """בודק הפלגה לנמל אחר"""
    ferry = Ferry("Test Ferry", 100, "Piraeus")
    
    # הפלגה תקינה
    time = ferry.sail_to("Santorini", 150, 30)
    assert time == 5.0  # 150 / 30 = 5
    assert ferry.current_port == "Santorini"
    
    # הפלגה עם מרחק/מהירות לא תקינים
    time = ferry.sail_to("Mykonos", -10, 30)
    assert time is None
    assert ferry.current_port == "Santorini"  # לא השתנה


if __name__ == "__main__":
    # הרצת הבדיקות ידנית (אם לא משתמשים ב-pytest)
    test_ferry_creation()
    print("✓ test_ferry_creation passed")
    
    test_load_passengers()
    print("✓ test_load_passengers passed")
    
    test_unload_passengers()
    print("✓ test_unload_passengers passed")
    
    test_sail_to()
    print("✓ test_sail_to passed")
    
    print("\nAll tests passed!")

