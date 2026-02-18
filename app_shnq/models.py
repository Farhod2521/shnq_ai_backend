import uuid
from django.db import connection, models


class Category(models.Model):
    """
    Normativ toifa: SHNQ, QMQ, SanQvaN
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    code = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.code


class Document(models.Model):
    """
    Hujjat: SHNQ 2.08.02-09 kabi
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    category = models.ForeignKey(Category, on_delete=models.PROTECT, related_name="documents")
    title = models.CharField(max_length=500)
    code = models.CharField(max_length=100, db_index=True)
    lex_url = models.URLField(blank=True, null=True)

    original_file = models.FileField(upload_to="docs/original/", blank=True, null=True)
    html_file = models.FileField(upload_to="docs/html/", blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("category", "code")

    def __str__(self):
        return self.code


class Chapter(models.Model):
    """
    Bob / bolim
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chapters")
    title = models.CharField(max_length=500)
    order = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.document.code} - {self.title}"


class Clause(models.Model):
    """
    Norma / band
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="clauses")
    chapter = models.ForeignKey(Chapter, on_delete=models.SET_NULL, null=True, related_name="clauses")

    clause_number = models.CharField(max_length=50, blank=True, null=True, db_index=True)
    html_anchor = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    text = models.TextField()

    order = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"{self.document.code} - {self.clause_number}"


class ClauseEmbedding(models.Model):
    """
    Embedding metadata va vector
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    clause = models.OneToOneField(Clause, on_delete=models.CASCADE, related_name="embedding")

    embedding_model = models.CharField(max_length=100)
    vector = models.JSONField()
    token_count = models.PositiveIntegerField(default=0)

    shnq_code = models.CharField(max_length=100)
    chapter_title = models.CharField(max_length=500, blank=True, null=True)
    clause_number = models.CharField(max_length=50, blank=True, null=True)
    lex_url = models.URLField(blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)


class NormTable(models.Model):
    """
    Hujjat ichidagi jadval
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="tables")
    chapter = models.ForeignKey(
        Chapter, on_delete=models.SET_NULL, null=True, blank=True, related_name="tables"
    )
    section_title = models.CharField(max_length=500, blank=True, null=True, db_index=True)
    table_number = models.CharField(max_length=50, db_index=True)
    title = models.CharField(max_length=500, blank=True, null=True)
    html_anchor = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    raw_html = models.TextField(blank=True, default="")
    raw_html_ru = models.TextField(blank=True, default="")
    raw_html_en = models.TextField(blank=True, default="")
    raw_html_ko = models.TextField(blank=True, default="")
    markdown = models.TextField(blank=True, default="")
    markdown_ru = models.TextField(blank=True, default="")
    markdown_en = models.TextField(blank=True, default="")
    markdown_ko = models.TextField(blank=True, default="")
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("document", "order")

    def __str__(self):
        return f"{self.document.code} - jadval {self.table_number}"


class NormTableRow(models.Model):
    """
    Jadval qatori
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    table = models.ForeignKey(NormTable, on_delete=models.CASCADE, related_name="rows")
    row_index = models.PositiveIntegerField()

    class Meta:
        unique_together = ("table", "row_index")
        ordering = ("table", "row_index")

    def __str__(self):
        return f"{self.table} row {self.row_index}"


class NormTableCell(models.Model):
    """
    Jadval katagi
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    row = models.ForeignKey(NormTableRow, on_delete=models.CASCADE, related_name="cells")
    col_index = models.PositiveIntegerField()
    text = models.TextField(blank=True, default="")
    is_header = models.BooleanField(default=False)
    row_span = models.PositiveIntegerField(default=1)
    col_span = models.PositiveIntegerField(default=1)

    class Meta:
        unique_together = ("row", "col_index")
        ordering = ("row", "col_index")

    def __str__(self):
        return f"{self.row} col {self.col_index}"


class TableRowEmbedding(models.Model):
    """
    Jadval satri bo'yicha embedding metadata va vector
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    row = models.OneToOneField(NormTableRow, on_delete=models.CASCADE, related_name="embedding")

    embedding_model = models.CharField(max_length=100)
    vector = models.JSONField()
    token_count = models.PositiveIntegerField(default=0)

    shnq_code = models.CharField(max_length=100)
    chapter_title = models.CharField(max_length=500, blank=True, null=True)
    table_number = models.CharField(max_length=50, db_index=True)
    table_title = models.CharField(max_length=500, blank=True, null=True)
    row_index = models.PositiveIntegerField(default=0)
    search_text = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)


class NormImage(models.Model):
    """
    Hujjat ichidagi rasm (img src)
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="images")
    chapter = models.ForeignKey(
        Chapter, on_delete=models.SET_NULL, null=True, blank=True, related_name="images"
    )
    section_title = models.CharField(max_length=500, blank=True, null=True, db_index=True)
    appendix_number = models.CharField(max_length=30, blank=True, null=True, db_index=True)
    title = models.CharField(max_length=500, blank=True, null=True)
    html_anchor = models.CharField(max_length=100, blank=True, null=True, db_index=True)
    image_url = models.URLField(max_length=1000)
    local_path = models.CharField(max_length=500, blank=True, default="")
    context_text = models.TextField(blank=True, default="")
    ocr_text = models.TextField(blank=True, default="")
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("document", "order")

    def __str__(self):
        return f"{self.document.code} - image {self.order}"


class ImageEmbedding(models.Model):
    """
    Rasm bo'yicha embedding metadata va vector
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.OneToOneField(NormImage, on_delete=models.CASCADE, related_name="embedding")

    embedding_model = models.CharField(max_length=100)
    vector = models.JSONField()
    token_count = models.PositiveIntegerField(default=0)

    shnq_code = models.CharField(max_length=100)
    chapter_title = models.CharField(max_length=500, blank=True, null=True)
    appendix_number = models.CharField(max_length=30, blank=True, null=True)
    image_url = models.URLField(max_length=1000)

    created_at = models.DateTimeField(auto_now_add=True)


class QuestionAnswer(models.Model):
    """
    RAG savol-javob logi
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.TextField()
    answer = models.TextField()
    top_clause_ids = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


def ensure_runtime_tables():
    """
    Migrations ishlatilmagan loyihalarda yangi jadvallarni runtime'da yaratish uchun.
    """
    existing_tables = set(connection.introspection.table_names())
    models_to_ensure = [NormImage, ImageEmbedding, TableRowEmbedding]

    with connection.schema_editor() as schema_editor:
        for model_cls in models_to_ensure:
            table_name = model_cls._meta.db_table
            if table_name in existing_tables:
                continue
            schema_editor.create_model(model_cls)
            existing_tables.add(table_name)
