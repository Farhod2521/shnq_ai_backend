import argparse
import os
import sys

import django


def main():
    parser = argparse.ArgumentParser(description="SHNQ matnlarini DeepSeek embeddingga otkazish.")
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding modeli (default: DEEPSEEK_EMBED_MODEL).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mavjud embeddinglarni ham qayta hisoblash.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nechta bandni qayta hisoblash (test uchun).",
    )
    parser.add_argument(
        "--only-images",
        action="store_true",
        help="Faqat rasmlar embeddingini qayta hisoblash.",
    )
    args = parser.parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()

    if args.only_images:
        from app_shnq.embeddings import upsert_image_embeddings

        image_result = upsert_image_embeddings(
            embedding_model=args.model,
            force_update=args.force,
            limit=args.limit,
        )
        result = {"clauses": {"created": 0, "updated": 0, "skipped": 0}, "images": image_result}
    else:
        from app_shnq.embeddings import upsert_all_embeddings

        result = upsert_all_embeddings(
            embedding_model=args.model,
            force_update=args.force,
            limit=args.limit,
        )

    print("Tayyor.")
    print(
        "Clauses -> "
        f"Created: {result['clauses']['created']}, "
        f"Updated: {result['clauses']['updated']}, "
        f"Skipped: {result['clauses']['skipped']}"
    )
    print(
        "Images  -> "
        f"Created: {result['images']['created']}, "
        f"Updated: {result['images']['updated']}, "
        f"Skipped: {result['images']['skipped']}"
    )


if __name__ == "__main__":
    sys.exit(main())
