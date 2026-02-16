import json
import random
import re
from pathlib import Path

from django.core.management.base import BaseCommand

from app_shnq.models import Clause


QUESTION_TEMPLATES = [
    "{doc} {clause} bandida asosiy talab nima?",
    "{doc} bo'yicha {clause} band mazmunini tushuntiring.",
    "{chapter} doirasida {clause} band nimalarni belgilaydi?",
    "{doc} dagi {clause} band bo'yicha amaliy talablari qaysilar?",
    "{clause} bandda xavfsizlik bo'yicha qanday me'yor berilgan?",
]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _make_question(clause, index: int) -> str:
    chapter = _clean_text(clause.chapter.title if clause.chapter else "Ushbu bo'lim")
    doc = _clean_text(clause.document.code if clause.document else "SHNQ")
    clause_num = _clean_text(clause.clause_number or "")
    if clause_num:
        clause_ref = f"{clause_num}"
    else:
        clause_ref = "mazkur"
    template = QUESTION_TEMPLATES[index % len(QUESTION_TEMPLATES)]
    return template.format(doc=doc, clause=clause_ref, chapter=chapter)


def _make_answer(clause) -> str:
    text = _clean_text(clause.text)
    if len(text) <= 700:
        return text
    return text[:700].rsplit(" ", 1)[0].strip() + " ..."


class Command(BaseCommand):
    help = "SHNQ bandlaridan few-shot savol-javob to'plamini yaratadi."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=50, help="Yaratiladigan QA soni.")
        parser.add_argument(
            "--doc-code",
            type=str,
            default="",
            help="Faqat bitta hujjat kodi bo'yicha tanlash (masalan: SHNQ 3.01.02-23).",
        )
        parser.add_argument(
            "--out",
            type=str,
            default="app_shnq/data/qa_fewshot.json",
            help="Natija JSON fayli yo'li.",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    def handle(self, *args, **options):
        limit = max(1, int(options["limit"]))
        doc_code = (options["doc_code"] or "").strip()
        out_path = Path(options["out"])
        random.seed(int(options["seed"]))

        qs = Clause.objects.select_related("document", "chapter").exclude(text__isnull=True).exclude(text__exact="")
        if doc_code:
            qs = qs.filter(document__code__iexact=doc_code)

        # Juda qisqa va foydasiz bandlarni olib tashlaymiz.
        clauses = [c for c in qs.order_by("order") if len(_clean_text(c.text)) >= 80]
        if not clauses:
            self.stdout.write(self.style.ERROR("Mos Clause topilmadi."))
            return

        with_number = [c for c in clauses if _clean_text(c.clause_number)]
        without_number = [c for c in clauses if not _clean_text(c.clause_number)]
        random.shuffle(with_number)
        random.shuffle(without_number)
        picked = (with_number + without_number)[:limit]

        items = []
        for idx, clause in enumerate(picked, 1):
            items.append(
                {
                    "id": idx,
                    "question": _make_question(clause, idx),
                    "answer": _make_answer(clause),
                    "shnq_code": _clean_text(clause.document.code if clause.document else ""),
                    "clause_number": _clean_text(clause.clause_number or ""),
                    "chapter": _clean_text(clause.chapter.title if clause.chapter else ""),
                }
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

        self.stdout.write(
            self.style.SUCCESS(f"Few-shot QA yaratildi: {len(items)} ta -> {out_path.as_posix()}")
        )
