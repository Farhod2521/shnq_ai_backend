from django.contrib import admin

from .models import (
    Category,
    Chapter,
    Clause,
    ClauseEmbedding,
    Document,
    ImageEmbedding,
    NormImage,
    NormTable,
    NormTableCell,
    NormTableRow,
    QuestionAnswer,
    TableRowEmbedding,
)


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ("code", "name")
    search_fields = ("code", "name")


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("code", "title", "category", "lex_url", "created_at")
    list_filter = ("category",)
    search_fields = ("code", "title")


@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ("title", "document", "order")
    list_filter = ("document",)
    search_fields = ("title",)
    ordering = ("document", "order")


@admin.register(Clause)
class ClauseAdmin(admin.ModelAdmin):
    list_display = ("clause_number", "document", "chapter", "order")
    list_filter = ("document", "chapter")
    search_fields = ("clause_number", "text", "html_anchor")
    ordering = ("document", "order")


@admin.register(ClauseEmbedding)
class ClauseEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("clause", "embedding_model", "token_count", "shnq_code")
    search_fields = ("clause__clause_number", "shnq_code", "chapter_title")


@admin.register(NormImage)
class NormImageAdmin(admin.ModelAdmin):
    list_display = ("document", "appendix_number", "title", "order", "html_anchor")
    list_filter = ("document", "appendix_number")
    search_fields = ("document__code", "title", "context_text", "image_url", "html_anchor")
    ordering = ("document", "order")


@admin.register(ImageEmbedding)
class ImageEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("image", "embedding_model", "token_count", "shnq_code")
    search_fields = ("image__document__code", "image__title", "image_url", "shnq_code")


@admin.register(TableRowEmbedding)
class TableRowEmbeddingAdmin(admin.ModelAdmin):
    list_display = ("row", "embedding_model", "row_index", "table_number", "shnq_code", "token_count")
    list_filter = ("shnq_code", "table_number")
    search_fields = ("shnq_code", "table_number", "table_title", "chapter_title", "search_text")
    ordering = ("shnq_code", "table_number", "row_index")


@admin.register(QuestionAnswer)
class QuestionAnswerAdmin(admin.ModelAdmin):
    list_display = ("question", "created_at")
    search_fields = ("question", "answer")


class NormTableCellInline(admin.TabularInline):
    model = NormTableCell
    extra = 0
    fields = ("col_index", "text", "is_header", "row_span", "col_span")
    ordering = ("col_index",)


@admin.register(NormTableRow)
class NormTableRowAdmin(admin.ModelAdmin):
    list_display = ("table", "row_index")
    list_filter = ("table__document",)
    search_fields = ("table__document__code", "table__table_number")
    ordering = ("table", "row_index")
    inlines = [NormTableCellInline]


@admin.register(NormTable)
class NormTableAdmin(admin.ModelAdmin):
    list_display = ("table_number", "document", "chapter", "order", "html_anchor")
    list_filter = ("document",)
    search_fields = ("table_number", "title", "document__code")
    ordering = ("document", "order")
