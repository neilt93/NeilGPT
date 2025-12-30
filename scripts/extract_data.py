"""
Data Extraction Script

Main entry point for extracting and preprocessing chat data.

Usage:
    python scripts/extract_data.py [--instagram-path PATH] [--skip-instagram]

This script:
1. Extracts iMessage data from SQLite database
2. Extracts Instagram data from export (if provided)
3. Applies content filtering (MANDATORY)
4. Preprocesses and formats data for training
5. Saves processed datasets
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.imessage_extractor import iMessageExtractor
from data.instagram_parser import InstagramParser
from data.preprocessor import ChatPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Extract and preprocess chat data for NeilGPT"
    )
    parser.add_argument(
        "--instagram-path",
        type=str,
        help="Path to Instagram data export (optional)"
    )
    parser.add_argument(
        "--skip-instagram",
        action="store_true",
        help="Skip Instagram data extraction"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=10,
        help="Maximum messages per conversation window (default: 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of messages to extract (for testing)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NEILGPT DATA EXTRACTION & PREPROCESSING")
    print("=" * 60)
    print()

    # Step 1: Extract iMessage data
    print("📱 STEP 1: Extracting iMessage data...")
    print("-" * 60)
    try:
        imessage_extractor = iMessageExtractor()
        imessage_messages = imessage_extractor.extract_messages(limit=args.limit)

        # Save iMessage data
        imessage_extractor.save_messages(imessage_messages)

        # Print stats
        stats = imessage_extractor.get_stats(imessage_messages)
        print(f"\n✓ Extracted {stats['total_messages']} iMessage messages")
        print(f"  - From you: {stats['messages_from_me']}")
        print(f"  - From others: {stats['messages_from_others']}")
        print(f"  - Date range: {stats['earliest_message']} to {stats['latest_message']}")

    except Exception as e:
        print(f"❌ Error extracting iMessage data: {e}")
        print("   Make sure you've granted Full Disk Access to your Terminal/IDE")
        print("   in System Preferences → Privacy & Security")
        sys.exit(1)

    # Step 2: Extract Instagram data (if provided)
    if not args.skip_instagram:
        print("\n📸 STEP 2: Extracting Instagram data...")
        print("-" * 60)

        if args.instagram_path:
            try:
                instagram_parser = InstagramParser(args.instagram_path)
                instagram_messages = instagram_parser.extract_all_messages()

                # Save Instagram data
                instagram_parser.save_messages(instagram_messages)

                # Print stats
                stats = instagram_parser.get_stats(instagram_messages)
                print(f"\n✓ Extracted {stats['total_messages']} Instagram messages")
                print(f"  - Unique conversations: {stats['unique_conversations']}")
                print(f"  - Date range: {stats['earliest_message']} to {stats['latest_message']}")

            except Exception as e:
                print(f"⚠️  Warning: Could not extract Instagram data: {e}")
                print("   Continuing with iMessage data only...")
        else:
            print("⏭️  No Instagram path provided. Skipping Instagram extraction.")
            print("   (Use --instagram-path to include Instagram data)")
    else:
        print("\n⏭️  STEP 2: Skipping Instagram data (--skip-instagram flag)")

    # Step 3: Preprocess data
    print("\n🔄 STEP 3: Preprocessing data...")
    print("-" * 60)
    try:
        preprocessor = ChatPreprocessor()
        preprocessor.process(max_context_length=args.max_context)

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Done!
    print("\n" + "=" * 60)
    print("✅ DATA EXTRACTION COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review filter report: data/processed/filter_report.json")
    print("  2. Check processed data: data/processed/dataset.json")
    print("  3. Proceed to tokenization and model training")
    print()


if __name__ == "__main__":
    main()
