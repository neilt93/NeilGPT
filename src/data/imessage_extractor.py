"""
iMessage Data Extractor

Extracts messages from macOS iMessage SQLite database (chat.db).
Handles individual and group conversations.

Database location: ~/Library/Messages/chat.db
Note: Requires "Full Disk Access" permission for Terminal/IDE in System Preferences.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm


class iMessageExtractor:
    # Mac absolute time reference (seconds since 2001-01-01)
    MAC_EPOCH = 978307200

    def __init__(self, db_path: Optional[str] = None, output_dir: str = "data/raw/imessage"):
        """
        Initialize iMessage extractor.

        Args:
            db_path: Path to chat.db file. Defaults to ~/Library/Messages/chat.db
            output_dir: Directory to save extracted messages
        """
        if db_path is None:
            db_path = os.path.expanduser("~/Library/Messages/chat.db")

        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"iMessage database not found at {self.db_path}. "
                "Make sure you're on macOS and have granted Full Disk Access."
            )

    def _mac_to_unix_timestamp(self, mac_time: int) -> int:
        """Convert Mac absolute time to Unix timestamp."""
        if mac_time is None:
            return None
        # Mac timestamps are in nanoseconds, convert to seconds
        return int(mac_time / 1000000000) + self.MAC_EPOCH

    def _format_timestamp(self, mac_time: int) -> str:
        """Convert Mac time to human-readable ISO format."""
        unix_time = self._mac_to_unix_timestamp(mac_time)
        if unix_time is None:
            return None
        return datetime.fromtimestamp(unix_time).isoformat()

    def extract_messages(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Extract all messages from iMessage database.

        Args:
            limit: Optional limit on number of messages to extract

        Returns:
            List of message dictionaries
        """
        print(f"Connecting to iMessage database: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query to extract messages with sender information
        query = """
        SELECT
            message.ROWID as message_id,
            message.text,
            message.date,
            message.is_from_me,
            message.cache_roomnames as chat_name,
            handle.id as sender_id,
            message.service as service
        FROM message
        LEFT JOIN handle ON message.handle_id = handle.ROWID
        WHERE message.text IS NOT NULL
            AND message.text != ''
        ORDER BY message.date ASC
        """

        if limit:
            query += f" LIMIT {limit}"

        print("Executing query...")
        cursor.execute(query)

        messages = []
        rows = cursor.fetchall()

        print(f"Extracting {len(rows)} messages...")
        for row in tqdm(rows):
            message_id, text, date, is_from_me, chat_name, sender_id, service = row

            messages.append({
                "message_id": message_id,
                "text": text,
                "timestamp": self._format_timestamp(date),
                "unix_timestamp": self._mac_to_unix_timestamp(date),
                "is_from_me": bool(is_from_me),
                "sender_id": sender_id if sender_id else "me",
                "chat_name": chat_name,
                "service": service,  # "iMessage" or "SMS"
                "source": "imessage"
            })

        conn.close()

        print(f"Extracted {len(messages)} messages")
        return messages

    def extract_conversations(self, limit: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
        Extract messages grouped by conversation.

        Args:
            limit: Optional limit on number of messages to extract

        Returns:
            Dictionary mapping chat_name/sender_id to list of messages
        """
        messages = self.extract_messages(limit=limit)

        conversations = {}
        for msg in messages:
            # Use chat_name for group chats, sender_id for individual chats
            key = msg.get("chat_name") or msg.get("sender_id") or "unknown"

            if key not in conversations:
                conversations[key] = []

            conversations[key].append(msg)

        print(f"Organized into {len(conversations)} conversations")
        return conversations

    def save_messages(self, messages: List[Dict], filename: str = "messages.json"):
        """Save extracted messages to JSON file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(messages)} messages to {output_path}")

    def save_conversations(self, conversations: Dict, filename: str = "conversations.json"):
        """Save conversations to JSON file."""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)

        total_messages = sum(len(msgs) for msgs in conversations.values())
        print(f"Saved {len(conversations)} conversations ({total_messages} messages) to {output_path}")

    def get_stats(self, messages: List[Dict]) -> Dict:
        """Get statistics about extracted messages."""
        from_me = sum(1 for msg in messages if msg["is_from_me"])
        from_others = len(messages) - from_me

        # Get date range
        timestamps = [msg["unix_timestamp"] for msg in messages if msg["unix_timestamp"]]
        if timestamps:
            earliest = datetime.fromtimestamp(min(timestamps))
            latest = datetime.fromtimestamp(max(timestamps))
        else:
            earliest = latest = None

        # Count services
        services = {}
        for msg in messages:
            service = msg.get("service", "unknown")
            services[service] = services.get(service, 0) + 1

        stats = {
            "total_messages": len(messages),
            "messages_from_me": from_me,
            "messages_from_others": from_others,
            "earliest_message": earliest.isoformat() if earliest else None,
            "latest_message": latest.isoformat() if latest else None,
            "services": services
        }

        return stats


def main():
    """Example usage of iMessageExtractor."""
    # Initialize extractor
    extractor = iMessageExtractor()

    # Extract all messages
    messages = extractor.extract_messages()

    # Get statistics
    stats = extractor.get_stats(messages)
    print("\n=== iMessage Statistics ===")
    print(f"Total messages: {stats['total_messages']}")
    print(f"From me: {stats['messages_from_me']}")
    print(f"From others: {stats['messages_from_others']}")
    print(f"Date range: {stats['earliest_message']} to {stats['latest_message']}")
    print(f"Services: {stats['services']}")

    # Save messages
    extractor.save_messages(messages)

    # Also save organized by conversation
    conversations = extractor.extract_conversations()
    extractor.save_conversations(conversations)

    print("\n✅ iMessage extraction complete!")


if __name__ == "__main__":
    main()
