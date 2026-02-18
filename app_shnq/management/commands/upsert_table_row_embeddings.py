from django.core.management.base import BaseCommand

from app_shnq.deepseek_client import DEFAULT_EMBED_MODEL
from app_shnq.table_embeddings import upsert_table_row_embeddings


class Command(BaseCommand):
    help = "NormTableRow satrlarini embeddingga tayyorlab TableRowEmbedding jadvaliga saqlaydi."

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            type=str,
            default=DEFAULT_EMBED_MODEL,
            help="Embedding modeli nomi.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Mavjud embeddinglar bo'lsa ham qayta hisoblash.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Nechta satrni qayta ishlash (0 = cheksiz).",
        )
        parser.add_argument(
            "--doc-code",
            type=str,
            default="",
            help="Faqat bitta hujjat kodi bo'yicha ishlash (masalan: SHNQ 2.07.01-23).",
        )
        parser.add_argument(
            "--table-number",
            type=str,
            default="",
            help="Faqat bitta jadval/ilova raqami bo'yicha ishlash (masalan: 7 yoki ilova-7).",
        )
        parser.add_argument(
            "--table-id",
            type=str,
            default="",
            help="Aniq jadval UUID bo'yicha ishlash.",
        )

    def handle(self, *args, **options):
        model = (options.get("model") or DEFAULT_EMBED_MODEL).strip()
        force = bool(options.get("force"))
        limit = int(options.get("limit") or 0)
        doc_code = (options.get("doc_code") or "").strip() or None
        table_number = (options.get("table_number") or "").strip() or None
        table_id = (options.get("table_id") or "").strip() or None

        result = upsert_table_row_embeddings(
            embedding_model=model,
            force_update=force,
            limit=limit if limit > 0 else None,
            doc_code=doc_code,
            table_number=table_number,
            table_id=table_id,
        )
        self.stdout.write(
            self.style.SUCCESS(
                "Done. Table rows -> created: {created}, updated: {updated}, skipped: {skipped}".format(**result)
            )
        )
