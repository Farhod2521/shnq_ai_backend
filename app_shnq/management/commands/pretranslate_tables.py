from django.core.management.base import BaseCommand
from django.db import models

from app_shnq.models import NormTable
from app_shnq.table_i18n import ensure_normtable_i18n_columns, pretranslate_table_fields


class Command(BaseCommand):
    help = "NormTable jadval HTML/markdown matnlarini ru/en/ko tillariga oldindan tarjima qilib saqlaydi."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Mavjud tarjimalarni ham qayta yaratadi.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Bir ishga tushirishda nechta jadvalni qayta ishlash (0 = cheksiz).",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Faqat bo'sh tarjimalar emas, hamma jadvallarni ko'rib chiqadi.",
        )

    def handle(self, *args, **options):
        force = bool(options.get("force"))
        limit = int(options.get("limit") or 0)
        process_all = bool(options.get("all"))
        ensure_normtable_i18n_columns()
        qs = NormTable.objects.all().order_by("document__code", "order")
        if not process_all and not force:
            qs = qs.filter(
                models.Q(raw_html_ru="")
                | models.Q(raw_html_en="")
                | models.Q(raw_html_ko="")
                | models.Q(markdown_ru="")
                | models.Q(markdown_en="")
                | models.Q(markdown_ko="")
            )
        total = qs.count()
        done = 0
        for table in qs.iterator():
            if limit > 0 and done >= limit:
                break
            pretranslate_table_fields(table, force=force)
            done += 1
            self.stdout.write(f"[{done}/{total}] {table.document.code} {table.table_number}")
        self.stdout.write(self.style.SUCCESS(f"Done. Tables processed: {done}"))
