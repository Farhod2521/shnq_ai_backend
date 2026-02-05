# shnq_ai_backend

## DeepSeek sozlamasi

PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="YOUR_TOKEN"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_CHAT_MODEL="deepseek-chat"
$env:DEEPSEEK_EMBED_MODEL="deepseek-embedding"
$env:DEEPSEEK_EMBED_FALLBACK="hash"
$env:RAG_REWRITE_QUERY="0"
$env:RAG_RERANK_ENABLED="0"
$env:RAG_EMBED_CACHE="1"
```

Embeddinglarni qayta hisoblash:

```powershell
python shnq_embding_llmma.py --force
```
