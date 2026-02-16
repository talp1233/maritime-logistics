"""
Notification service — sends sailing ban alerts via Telegram.

Setup:
  1. Create a bot via @BotFather on Telegram -> get BOT_TOKEN
  2. Start a chat with your bot and get CHAT_ID
  3. Set environment variables:
       export TELEGRAM_BOT_TOKEN="your-token"
       export TELEGRAM_CHAT_ID="your-chat-id"
  4. Run: python -m src.services.notifications --test
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TelegramNotifier:
    """Sends sailing ban alerts to a Telegram chat."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._base_url = f"https://api.telegram.org/bot{self.bot_token}"

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send a text message to the configured chat."""
        if not self.is_configured:
            logger.warning("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
            return False

        url = f"{self._base_url}/sendMessage"
        data = json.dumps({
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("ok"):
                    logger.info("Telegram message sent successfully")
                    return True
                logger.error("Telegram API error: %s", result)
                return False
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", e)
            return False

    def format_route_alert(self, route_result: dict) -> str:
        """Format a route check result as a Telegram HTML message."""
        status = route_result["overall_status"]
        route_name = route_result["route_name"]
        route_id = route_result["route_id"]

        # Status icon
        icons = {"BAN_LIKELY": "🔴", "AT_RISK": "🟡", "CLEAR": "🟢"}
        icon = icons.get(status, "⚪")

        # Find worst conditions
        hourly = route_result.get("hourly", [])
        max_wind = max((h["wind_speed_knots"] for h in hourly), default=0)
        max_wave = max((h["wave_height_m"] for h in hourly), default=0)
        max_bf = max((h["beaufort"] for h in hourly), default=0)

        return (
            f"{icon} <b>{route_name}</b> ({route_id})\n"
            f"Status: <b>{status}</b>\n"
            f"Max wind: {max_wind:.0f} kn (Beaufort {max_bf})\n"
            f"Max wave: {max_wave:.1f} m\n"
            f"Vessel: {route_result.get('vessel_type', 'CONVENTIONAL')}"
        )

    def send_summary(self, results: list[dict]) -> bool:
        """Send a summary of all route checks."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        ban_routes = [r for r in results if r["overall_status"] == "BAN_LIKELY"]
        risk_routes = [r for r in results if r["overall_status"] == "AT_RISK"]
        clear_routes = [r for r in results if r["overall_status"] == "CLEAR"]

        lines = [
            f"<b>⚓ Maritime Weather Report</b>",
            f"<i>{now}</i>",
            "",
        ]

        if ban_routes:
            lines.append(f"🔴 <b>BAN LIKELY ({len(ban_routes)} routes):</b>")
            for r in ban_routes:
                lines.append(f"  • {r['route_name']}")
            lines.append("")

        if risk_routes:
            lines.append(f"🟡 <b>AT RISK ({len(risk_routes)} routes):</b>")
            for r in risk_routes:
                lines.append(f"  • {r['route_name']}")
            lines.append("")

        lines.append(f"🟢 Clear: {len(clear_routes)} routes")
        lines.append(f"\nTotal: {len(results)} routes checked")

        return self.send_message("\n".join(lines))

    def send_alert_if_needed(self, results: list[dict]) -> bool:
        """Only send a notification if there are BAN_LIKELY or AT_RISK routes."""
        has_alerts = any(
            r["overall_status"] in ("BAN_LIKELY", "AT_RISK")
            for r in results
        )
        if has_alerts:
            return self.send_summary(results)
        logger.info("All routes clear — no alert sent.")
        return False


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Telegram notifications")
    parser.add_argument("--test", action="store_true", help="Send a test message")
    args = parser.parse_args()

    if args.test:
        notifier = TelegramNotifier()
        if not notifier.is_configured:
            print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.")
        else:
            success = notifier.send_message(
                "⚓ <b>Maritime Intelligence Platform</b>\n"
                "Test notification — system is working!"
            )
            print("Sent!" if success else "Failed.")
