"""
Content Filter (MANDATORY for Professional Project)

Filters PII (Personally Identifiable Information) and sensitive content from chat data.
This ensures the trained model is safe for public demonstration and portfolio use.

Filtering includes:
- PII: Phone numbers, emails, names, addresses, credit cards, SSNs
- Financial information: Banking details, transactions, salaries
- Sensitive topics: Health details, specific political info, etc.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json


# Only import spaCy if available (for NER)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not installed. Name detection will use simple heuristics.")


class ContentFilter:
    def __init__(self, config_path: str = "config/filter_config.yaml"):
        """
        Initialize content filter with configuration.

        Args:
            config_path: Path to filter configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize spaCy NER if available and enabled
        self.nlp = None
        if SPACY_AVAILABLE and self.config.get("pii_filtering", {}).get("names", {}).get("use_ner", True):
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✓ Loaded spaCy NER model for name detection")
            except OSError:
                print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")

        # Statistics tracking
        self.stats = {
            "total_messages": 0,
            "messages_filtered": 0,
            "messages_kept": 0,
            "pii_instances_removed": 0,
            "financial_messages_removed": 0,
            "sensitive_messages_removed": 0,
            "filtered_messages": []  # Store filtered messages for review
        }

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for PII detection."""
        # Phone number patterns (US and international)
        self.phone_pattern = re.compile(
            r'\b(?:\+?1[-.]?)?'  # Optional +1 or 1
            r'(?:\(\d{3}\)|\d{3})[-.\s]?'  # Area code
            r'\d{3}[-.\s]?'  # First 3 digits
            r'\d{4}\b'  # Last 4 digits
        )

        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # Credit card pattern (13-16 digits)
        self.card_pattern = re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'  # 16 digit cards
            r'|\b(?:\d{4}[-\s]?){2}\d{5}\b'  # 15 digit cards (Amex)
        )

        # SSN pattern
        self.ssn_pattern = re.compile(
            r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        )

        # Address pattern (basic - street numbers + street names)
        self.address_pattern = re.compile(
            r'\b\d{1,5}\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b',
            re.IGNORECASE
        )

        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        # Dollar amount pattern (for financial filtering)
        self.dollar_pattern = re.compile(r'\$\s?[\d,]+(?:\.\d{2})?')

    def _detect_names(self, text: str) -> List[str]:
        """
        Detect person names in text using spaCy NER.

        Args:
            text: Input text

        Returns:
            List of detected names
        """
        if self.nlp is None:
            # Fallback: Simple capitalized word heuristic
            # This is very basic and will have false positives
            words = text.split()
            return [w for w in words if w.istitle() and len(w) > 2]

        doc = self.nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    def _filter_pii(self, text: str) -> Tuple[str, int]:
        """
        Filter PII from text.

        Args:
            text: Input text

        Returns:
            Tuple of (filtered_text, num_pii_instances_removed)
        """
        if not self.config.get("pii_filtering", {}).get("enabled", True):
            return text, 0

        pii_config = self.config["pii_filtering"]
        pii_count = 0
        filtered_text = text

        # Phone numbers
        if pii_config.get("phone_numbers", {}).get("enabled", True):
            phone_matches = len(self.phone_pattern.findall(filtered_text))
            if phone_matches > 0:
                token = pii_config["phone_numbers"].get("replacement_token", "<PHONE>")
                filtered_text = self.phone_pattern.sub(token, filtered_text)
                pii_count += phone_matches

        # Email addresses
        if pii_config.get("email_addresses", {}).get("enabled", True):
            email_matches = len(self.email_pattern.findall(filtered_text))
            if email_matches > 0:
                token = pii_config["email_addresses"].get("replacement_token", "<EMAIL>")
                filtered_text = self.email_pattern.sub(token, filtered_text)
                pii_count += email_matches

        # Names (NER-based)
        if pii_config.get("names", {}).get("enabled", True):
            names = self._detect_names(filtered_text)
            if names:
                token = pii_config["names"].get("replacement_token", "<NAME>")
                for name in names:
                    filtered_text = filtered_text.replace(name, token)
                    pii_count += 1

        # Addresses
        if pii_config.get("addresses", {}).get("enabled", True):
            address_matches = len(self.address_pattern.findall(filtered_text))
            if address_matches > 0:
                token = pii_config["addresses"].get("replacement_token", "<ADDRESS>")
                filtered_text = self.address_pattern.sub(token, filtered_text)
                pii_count += address_matches

        # Credit cards
        if pii_config.get("credit_cards", {}).get("enabled", True):
            card_matches = len(self.card_pattern.findall(filtered_text))
            if card_matches > 0:
                token = pii_config["credit_cards"].get("replacement_token", "<CARD>")
                filtered_text = self.card_pattern.sub(token, filtered_text)
                pii_count += card_matches

        # SSNs
        if pii_config.get("ssn", {}).get("enabled", True):
            ssn_matches = len(self.ssn_pattern.findall(filtered_text))
            if ssn_matches > 0:
                token = pii_config["ssn"].get("replacement_token", "<SSN>")
                filtered_text = self.ssn_pattern.sub(token, filtered_text)
                pii_count += ssn_matches

        return filtered_text, pii_count

    def _contains_financial_keywords(self, text: str) -> bool:
        """Check if text contains financial keywords."""
        if not self.config.get("financial_filtering", {}).get("enabled", True):
            return False

        keywords = self.config["financial_filtering"].get("keywords", [])
        text_lower = text.lower()

        return any(keyword.lower() in text_lower for keyword in keywords)

    def _contains_sensitive_keywords(self, text: str) -> bool:
        """Check if text contains sensitive topic keywords that should be removed."""
        if not self.config.get("sensitive_topics", {}).get("enabled", True):
            return False

        sensitive_config = self.config["sensitive_topics"]
        text_lower = text.lower()

        # Check health keywords
        if sensitive_config.get("health", {}).get("enabled", True):
            health_keywords = sensitive_config["health"].get("keywords", [])
            if any(keyword.lower() in text_lower for keyword in health_keywords):
                return True

        # Check political keywords (that indicate specific details to remove)
        if sensitive_config.get("politics", {}).get("enabled", True):
            politics_keywords = sensitive_config["politics"].get("keywords_to_remove", [])
            if any(keyword.lower() in text_lower for keyword in politics_keywords):
                return True

        return False

    def _filter_urls(self, text: str) -> str:
        """Filter URLs from text."""
        if not self.config.get("urls", {}).get("enabled", True):
            return text

        token = self.config["urls"].get("replacement_token")
        if token is None:
            # Remove URLs entirely
            return self.url_pattern.sub('', text)
        else:
            # Replace with token
            return self.url_pattern.sub(token, text)

    def _should_skip_message(self, text: str) -> bool:
        """Check if message matches skip patterns (reactions, system messages)."""
        skip_patterns = self.config.get("skip_patterns", [])

        for pattern in skip_patterns:
            if re.match(pattern, text):
                return True

        return False

    def filter_message(self, message: Dict) -> Optional[Dict]:
        """
        Filter a single message.

        Args:
            message: Message dictionary with 'text' field

        Returns:
            Filtered message dict or None if message should be removed entirely
        """
        self.stats["total_messages"] += 1

        text = message.get("text", "")

        # Check minimum length
        min_length = self.config.get("min_message_length", 2)
        if len(text) < min_length:
            self.stats["messages_filtered"] += 1
            return None

        # Check skip patterns
        if self._should_skip_message(text):
            self.stats["messages_filtered"] += 1
            return None

        # Check for financial keywords (remove entire message)
        if self._contains_financial_keywords(text):
            self.stats["financial_messages_removed"] += 1
            self.stats["messages_filtered"] += 1
            self.stats["filtered_messages"].append({
                "text": text,
                "reason": "financial",
                "timestamp": message.get("timestamp")
            })
            return None

        # Check for sensitive keywords (remove entire message)
        if self._contains_sensitive_keywords(text):
            self.stats["sensitive_messages_removed"] += 1
            self.stats["messages_filtered"] += 1
            self.stats["filtered_messages"].append({
                "text": text,
                "reason": "sensitive",
                "timestamp": message.get("timestamp")
            })
            return None

        # Filter PII
        filtered_text, pii_count = self._filter_pii(text)
        self.stats["pii_instances_removed"] += pii_count

        # Filter URLs
        filtered_text = self._filter_urls(filtered_text)

        # Update message
        filtered_message = message.copy()
        filtered_message["text"] = filtered_text
        filtered_message["pii_filtered"] = pii_count > 0

        self.stats["messages_kept"] += 1
        return filtered_message

    def filter_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Filter a list of messages.

        Args:
            messages: List of message dictionaries

        Returns:
            List of filtered messages (some may be removed)
        """
        filtered = []

        for msg in messages:
            filtered_msg = self.filter_message(msg)
            if filtered_msg is not None:
                filtered.append(filtered_msg)

        return filtered

    def save_report(self, output_path: str = "data/processed/filter_report.json"):
        """Save filtering statistics report."""
        report_path = Path(output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate percentages
        if self.stats["total_messages"] > 0:
            kept_pct = (self.stats["messages_kept"] / self.stats["total_messages"]) * 100
            filtered_pct = (self.stats["messages_filtered"] / self.stats["total_messages"]) * 100
        else:
            kept_pct = filtered_pct = 0

        report = {
            "total_messages_processed": self.stats["total_messages"],
            "messages_kept": self.stats["messages_kept"],
            "messages_filtered": self.stats["messages_filtered"],
            "kept_percentage": round(kept_pct, 2),
            "filtered_percentage": round(filtered_pct, 2),
            "pii_instances_removed": self.stats["pii_instances_removed"],
            "financial_messages_removed": self.stats["financial_messages_removed"],
            "sensitive_messages_removed": self.stats["sensitive_messages_removed"]
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n=== Content Filtering Report ===")
        print(f"Total messages processed: {report['total_messages_processed']}")
        print(f"Messages kept: {report['messages_kept']} ({report['kept_percentage']}%)")
        print(f"Messages filtered: {report['messages_filtered']} ({report['filtered_percentage']}%)")
        print(f"PII instances removed: {report['pii_instances_removed']}")
        print(f"Financial messages removed: {report['financial_messages_removed']}")
        print(f"Sensitive messages removed: {report['sensitive_messages_removed']}")
        print(f"\nReport saved to: {report_path}")

        # Optionally save filtered messages for manual review
        if self.config.get("logging", {}).get("save_filtered_messages", True):
            filtered_path = self.config["logging"].get(
                "filtered_messages_path",
                "data/processed/filtered_messages.json"
            )
            with open(filtered_path, 'w') as f:
                json.dump(self.stats["filtered_messages"], f, indent=2)
            print(f"Filtered messages saved to: {filtered_path} (for manual review)")

    def get_stats(self) -> Dict:
        """Get current filtering statistics."""
        return self.stats.copy()


def main():
    """Example usage of ContentFilter."""
    # Load sample messages
    sample_messages = [
        {"text": "Hey, my number is 555-123-4567", "timestamp": "2024-01-01"},
        {"text": "Email me at john.doe@example.com", "timestamp": "2024-01-02"},
        {"text": "I paid $5000 for that car", "timestamp": "2024-01-03"},
        {"text": "Just a normal message", "timestamp": "2024-01-04"},
        {"text": "I was diagnosed with XYZ", "timestamp": "2024-01-05"},
        {"text": "Liked a message", "timestamp": "2024-01-06"},
    ]

    # Initialize filter
    filter = ContentFilter()

    # Filter messages
    filtered_messages = filter.filter_messages(sample_messages)

    print("\n=== Original vs Filtered ===")
    for orig, filt in zip(sample_messages, filtered_messages):
        if filt:
            print(f"Original: {orig['text']}")
            print(f"Filtered: {filt['text']}")
            print()

    # Save report
    filter.save_report()

    print("\n✅ Content filtering complete!")


if __name__ == "__main__":
    main()
