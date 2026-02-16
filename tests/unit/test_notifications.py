"""Tests for notification service (no actual Telegram calls)."""

from src.services.notifications import TelegramNotifier


class TestTelegramNotifier:
    def test_not_configured_by_default(self):
        notifier = TelegramNotifier()
        assert not notifier.is_configured

    def test_configured_with_params(self):
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        assert notifier.is_configured

    def test_send_returns_false_when_not_configured(self):
        notifier = TelegramNotifier()
        assert notifier.send_message("test") is False

    def test_format_route_alert(self):
        notifier = TelegramNotifier()
        result = {
            "route_id": "PIR-MYK",
            "route_name": "Piraeus -> Mykonos",
            "overall_status": "BAN_LIKELY",
            "vessel_type": "CONVENTIONAL",
            "hourly": [
                {"wind_speed_knots": 35, "wave_height_m": 4.0, "beaufort": 8},
                {"wind_speed_knots": 40, "wave_height_m": 5.0, "beaufort": 8},
            ],
        }
        msg = notifier.format_route_alert(result)
        assert "PIR-MYK" in msg
        assert "BAN_LIKELY" in msg
        assert "40" in msg  # max wind

    def test_send_summary_format(self):
        notifier = TelegramNotifier()
        results = [
            {"route_id": "PIR-MYK", "route_name": "Piraeus -> Mykonos",
             "overall_status": "BAN_LIKELY", "hourly": []},
            {"route_id": "PIR-SAN", "route_name": "Piraeus -> Santorini",
             "overall_status": "CLEAR", "hourly": []},
        ]
        # Should not crash even when not configured
        sent = notifier.send_summary(results)
        assert sent is False  # Not configured

    def test_alert_if_needed_no_alerts(self):
        notifier = TelegramNotifier()
        results = [
            {"overall_status": "CLEAR"},
            {"overall_status": "CLEAR"},
        ]
        assert notifier.send_alert_if_needed(results) is False
