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
    args = parser.parse_args()

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    django.setup()

    from app_shnq.embeddings import upsert_clause_embeddings

    result = upsert_clause_embeddings(
        embedding_model=args.model,
        force_update=args.force,
        limit=args.limit,
    )

    print("Tayyor.")
    print(
        f"Created: {result['created']}, Updated: {result['updated']}, Skipped: {result['skipped']}"
    )


if __name__ == "__main__":
    sys.exit(main())
