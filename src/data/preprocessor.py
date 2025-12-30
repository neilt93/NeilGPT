"""
Data Preprocessor

Combines and preprocesses chat data from multiple sources (iMessage + Instagram).
Applies content filtering, text cleaning, and formats data for training.

Pipeline:
1. Load data from iMessage and Instagram
2. Apply content filtering (MANDATORY)
3. Clean and normalize text
4. Format into conversation sequences
5. Split into train/val/test sets
6. Save processed data
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
import random

from .content_filter import ContentFilter


class ChatPreprocessor:
    # Special tokens for sequence formatting
    BOS_TOKEN = "<BOS>"   # Beginning of sequence
    EOS_TOKEN = "<EOS>"   # End of sequence
    SEP_TOKEN = "<SEP>"   # Separator between messages
    PAD_TOKEN = "<PAD>"   # Padding token
    UNK_TOKEN = "<UNK>"   # Unknown token

    def __init__(self,
                 filter_config_path: str = "config/filter_config.yaml",
                 output_dir: str = "data/processed"):
        """
        Initialize preprocessor.

        Args:
            filter_config_path: Path to content filter configuration
            output_dir: Directory to save processed data
        """
        self.content_filter = ContentFilter(filter_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_imessage_data(self, path: str = "data/raw/imessage/messages.json") -> List[Dict]:
        """Load iMessage data."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_instagram_data(self, path: str = "data/raw/instagram/messages.json") -> List[Dict]:
        """Load Instagram data."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Instagram data not found at {path}. Skipping...")
            return []

    def combine_data_sources(self) -> List[Dict]:
        """Combine iMessage and Instagram data into unified format."""
        print("Loading data sources...")

        imessage_msgs = self.load_imessage_data()
        instagram_msgs = self.load_instagram_data()

        print(f"Loaded {len(imessage_msgs)} iMessage messages")
        print(f"Loaded {len(instagram_msgs)} Instagram messages")

        # Combine and sort by timestamp
        all_messages = imessage_msgs + instagram_msgs
        all_messages.sort(key=lambda x: x.get("unix_timestamp", 0))

        print(f"Combined total: {len(all_messages)} messages")
        return all_messages

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Keep emojis (they're contextually important!)
        # No need to remove them

        return text

    def filter_and_clean_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Apply content filtering and text cleaning to all messages.

        Args:
            messages: List of messages

        Returns:
            Filtered and cleaned messages
        """
        print("\n🔒 Applying content filtering (PII removal, sensitive content)...")

        # Apply content filter (MANDATORY)
        filtered_messages = self.content_filter.filter_messages(messages)

        # Clean remaining messages
        print("Cleaning text...")
        for msg in tqdm(filtered_messages):
            msg["text"] = self.clean_text(msg["text"])

        # Save filter report
        self.content_filter.save_report()

        return filtered_messages

    def group_into_conversations(self, messages: List[Dict],
                                  max_context_length: int = 10) -> List[List[Dict]]:
        """
        Group messages into conversation windows.

        Args:
            messages: List of messages
            max_context_length: Maximum number of messages per conversation window

        Returns:
            List of conversation windows
        """
        # Group by conversation ID (sender for DMs, chat_name for groups)
        conversations = {}

        for msg in messages:
            # Create conversation key
            if msg["source"] == "imessage":
                conv_key = msg.get("chat_name") or msg.get("sender_id", "unknown")
            else:  # Instagram
                conv_key = msg.get("conversation_name", "unknown")

            if conv_key not in conversations:
                conversations[conv_key] = []

            conversations[conv_key].append(msg)

        # Create sliding windows
        windows = []
        for conv_messages in conversations.values():
            # Skip very short conversations
            if len(conv_messages) < 2:
                continue

            # Create windows
            for i in range(0, len(conv_messages), max_context_length // 2):
                window = conv_messages[i:i + max_context_length]
                if len(window) >= 2:  # Need at least 2 messages for context
                    windows.append(window)

        return windows

    def format_for_training(self, conversations: List[List[Dict]]) -> List[Dict]:
        """
        Format conversation windows into training examples.

        Uses autoregressive format:
        Input: <BOS> msg1 <SEP> msg2 <SEP> msg3
        Target: msg1 <SEP> msg2 <SEP> msg3 <EOS> (shifted by 1 token)

        Args:
            conversations: List of conversation windows

        Returns:
            List of training examples
        """
        training_examples = []

        for conversation in conversations:
            # Build conversation sequence
            sequence_parts = []

            for msg in conversation:
                text = msg["text"]
                # Optionally add sender prefix for multi-party conversations
                # For single-party (your messages), we skip this
                sequence_parts.append(text)

            # Join with separator
            sequence = f" {self.SEP_TOKEN} ".join(sequence_parts)

            # Add BOS and EOS
            full_sequence = f"{self.BOS_TOKEN} {sequence} {self.EOS_TOKEN}"

            training_examples.append({
                "text": full_sequence,
                "num_messages": len(conversation),
                "conversation_start": conversation[0].get("timestamp"),
                "conversation_end": conversation[-1].get("timestamp")
            })

        return training_examples

    def split_dataset(self, examples: List[Dict],
                      train_ratio: float = 0.9,
                      val_ratio: float = 0.05,
                      test_ratio: float = 0.05) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split data into train/validation/test sets.

        Args:
            examples: List of training examples
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing

        Returns:
            Tuple of (train, val, test) lists
        """
        # Shuffle for random split
        random.shuffle(examples)

        total = len(examples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_set = examples[:train_end]
        val_set = examples[train_end:val_end]
        test_set = examples[val_end:]

        print(f"\nDataset split:")
        print(f"  Train: {len(train_set)} examples ({train_ratio * 100}%)")
        print(f"  Val:   {len(val_set)} examples ({val_ratio * 100}%)")
        print(f"  Test:  {len(test_set)} examples ({test_ratio * 100}%)")

        return train_set, val_set, test_set

    def save_processed_data(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """Save processed datasets to JSON files."""
        # Save each split
        splits = {"train": train, "val": val, "test": test}

        for split_name, data in splits.items():
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(data)} examples to {output_path}")

        # Save combined dataset for reference
        all_data = {
            "train": train,
            "val": val,
            "test": test,
            "metadata": {
                "total_examples": len(train) + len(val) + len(test),
                "train_examples": len(train),
                "val_examples": len(val),
                "test_examples": len(test),
                "special_tokens": {
                    "bos": self.BOS_TOKEN,
                    "eos": self.EOS_TOKEN,
                    "sep": self.SEP_TOKEN,
                    "pad": self.PAD_TOKEN,
                    "unk": self.UNK_TOKEN
                },
                "created_at": datetime.now().isoformat()
            }
        }

        combined_path = self.output_dir / "dataset.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"Saved combined dataset to {combined_path}")

    def process(self, max_context_length: int = 10):
        """
        Run complete preprocessing pipeline.

        Args:
            max_context_length: Maximum messages per conversation window
        """
        print("="* 50)
        print("CHAT DATA PREPROCESSING PIPELINE")
        print("=" * 50)

        # 1. Combine data sources
        messages = self.combine_data_sources()

        # 2. Filter and clean (MANDATORY step)
        filtered_messages = self.filter_and_clean_messages(messages)

        # 3. Group into conversations
        print(f"\nGrouping into conversation windows (max {max_context_length} messages)...")
        conversations = self.group_into_conversations(filtered_messages, max_context_length)
        print(f"Created {len(conversations)} conversation windows")

        # 4. Format for training
        print("\nFormatting for training...")
        training_examples = self.format_for_training(conversations)
        print(f"Generated {len(training_examples)} training examples")

        # 5. Split dataset
        train, val, test = self.split_dataset(training_examples)

        # 6. Save processed data
        print("\nSaving processed datasets...")
        self.save_processed_data(train, val, test)

        print("\n✅ Preprocessing complete!")
        print(f"📁 Processed data saved to: {self.output_dir}")


def main():
    """Example usage of ChatPreprocessor."""
    preprocessor = ChatPreprocessor()
    preprocessor.process(max_context_length=10)


if __name__ == "__main__":
    main()
