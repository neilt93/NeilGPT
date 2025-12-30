"""
Instagram Data Parser

Parses Instagram data exports (JSON format) to extract direct messages.

Instagram exports messages in JSON format within the `messages/inbox/` directory.
Each conversation is in a separate folder with a `message_1.json` file.

To get your Instagram data:
1. Go to Instagram Settings → Privacy and Security → Download Your Information
2. Select "Messages" and download as JSON
3. Extract the zip file
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm


class InstagramParser:
    def __init__(self, instagram_export_path: str, output_dir: str = "data/raw/instagram"):
        """
        Initialize Instagram parser.

        Args:
            instagram_export_path: Path to extracted Instagram data export
            output_dir: Directory to save parsed messages
        """
        self.export_path = Path(instagram_export_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Look for messages/inbox directory
        self.inbox_path = self.export_path / "messages" / "inbox"

        if not self.inbox_path.exists():
            # Try alternative path (sometimes it's in your_instagram_activity/)
            alt_path = self.export_path / "your_instagram_activity" / "messages" / "inbox"
            if alt_path.exists():
                self.inbox_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Could not find Instagram messages at {self.inbox_path}. "
                    f"Make sure you've extracted the Instagram data export correctly."
                )

    def _decode_instagram_text(self, text: str) -> str:
        """
        Decode Instagram's UTF-8 encoding.

        Instagram sometimes encodes text in a weird way. This fixes it.
        """
        try:
            # Instagram uses latin1 encoding that needs to be decoded as UTF-8
            return text.encode('latin1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            # If decoding fails, return original text
            return text

    def _parse_conversation(self, conversation_path: Path) -> List[Dict]:
        """
        Parse a single conversation folder.

        Args:
            conversation_path: Path to conversation folder

        Returns:
            List of messages from this conversation
        """
        messages = []

        # Instagram splits large conversations into multiple files
        message_files = sorted(conversation_path.glob("message_*.json"))

        for message_file in message_files:
            try:
                with open(message_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                conversation_name = data.get("title", "unknown")
                participants = data.get("participants", [])

                for msg in data.get("messages", []):
                    # Only include text messages (skip photos, videos, etc.)
                    if "content" not in msg:
                        continue

                    text = self._decode_instagram_text(msg["content"])

                    # Convert timestamp (milliseconds since epoch)
                    timestamp_ms = msg.get("timestamp_ms", 0)
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()

                    sender = self._decode_instagram_text(msg.get("sender_name", "unknown"))

                    messages.append({
                        "text": text,
                        "timestamp": timestamp,
                        "unix_timestamp": int(timestamp_ms / 1000),
                        "sender": sender,
                        "conversation_name": conversation_name,
                        "participants": [self._decode_instagram_text(p.get("name", "")) for p in participants],
                        "source": "instagram"
                    })

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {message_file}: {e}")
                continue

        return messages

    def extract_all_messages(self) -> List[Dict]:
        """
        Extract all messages from all conversations.

        Returns:
            List of all messages from all conversations
        """
        all_messages = []

        # Get all conversation folders
        conversation_folders = [d for d in self.inbox_path.iterdir() if d.is_dir()]

        print(f"Found {len(conversation_folders)} conversations")

        for conversation_folder in tqdm(conversation_folders, desc="Parsing conversations"):
            messages = self._parse_conversation(conversation_folder)
            all_messages.extend(messages)

        # Sort by timestamp
        all_messages.sort(key=lambda x: x["unix_timestamp"])

        print(f"Extracted {len(all_messages)} total messages")
        return all_messages

    def extract_conversations(self) -> Dict[str, List[Dict]]:
        """
        Extract messages grouped by conversation.

        Returns:
            Dictionary mapping conversation name to list of messages
        """
        all_messages = self.extract_all_messages()

        conversations = {}
        for msg in all_messages:
            conv_name = msg["conversation_name"]

            if conv_name not in conversations:
                conversations[conv_name] = []

            conversations[conv_name].append(msg)

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
        # Get date range
        timestamps = [msg["unix_timestamp"] for msg in messages]
        if timestamps:
            earliest = datetime.fromtimestamp(min(timestamps))
            latest = datetime.fromtimestamp(max(timestamps))
        else:
            earliest = latest = None

        # Count unique conversations
        conversations = set(msg["conversation_name"] for msg in messages)

        # Count unique senders
        senders = set(msg["sender"] for msg in messages)

        stats = {
            "total_messages": len(messages),
            "unique_conversations": len(conversations),
            "unique_senders": len(senders),
            "earliest_message": earliest.isoformat() if earliest else None,
            "latest_message": latest.isoformat() if latest else None
        }

        return stats


def main():
    """Example usage of InstagramParser."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python instagram_parser.py <path_to_instagram_export>")
        print("\nExample:")
        print("  python instagram_parser.py ~/Downloads/instagram-export")
        sys.exit(1)

    export_path = sys.argv[1]

    # Initialize parser
    parser = InstagramParser(export_path)

    # Extract all messages
    messages = parser.extract_all_messages()

    # Get statistics
    stats = parser.get_stats(messages)
    print("\n=== Instagram Statistics ===")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Unique conversations: {stats['unique_conversations']}")
    print(f"Unique senders: {stats['unique_senders']}")
    print(f"Date range: {stats['earliest_message']} to {stats['latest_message']}")

    # Save messages
    parser.save_messages(messages)

    # Also save organized by conversation
    conversations = parser.extract_conversations()
    parser.save_conversations(conversations)

    print("\n✅ Instagram parsing complete!")


if __name__ == "__main__":
    main()
