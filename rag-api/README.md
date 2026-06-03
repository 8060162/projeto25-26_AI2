# RAG API — Módulo de Acesso e Gestão de Credenciais

Camada de entrada protegida para o sistema de Retrieval-Augmented Generation (RAG) interno do Instituto Politécnico do Porto. Este módulo disponibiliza mecanismos de autenticação, autorização e gestão de aplicações clientes sobre o pipeline de recuperação e geração de respostas existente.

---

## Pré-requisitos

- Python 3.10 ou superior
- MongoDB Atlas (plano gratuito suficiente para a fase de desenvolvimento)
- Redis (instalação local)
- Repositório PROJETO25-26_AI2 na pasta imediatamente superior

---

## Instalação

```bash
# A partir da raiz do projecto
cd PROJETO25-26_AI2/rag-api

# Instalar as dependências
pip install -r requirements.txt

# Copiar o ficheiro de configuração e preencher as credenciais
cp .env.example .env
```

---

## Configuração — ficheiro `.env`

```bash
# MongoDB Atlas
MONGODB_URL=mongodb+srv://utilizador:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=ragdb

# Redis (local)
REDIS_URL=redis://localhost:6379/0
AUTH_CACHE_TTL_SECONDS=300
RATE_LIMIT_WINDOW_SECONDS=60
SESSION_TTL_SECONDS=1800

# Autenticação
API_KEY_PREFIX=rag
API_KEY_ENTROPY_BYTES=32
DEFAULT_RATE_LIMIT=100

# Pipeline
CHROMA_API_KEY=           # solicitar ao responsável pelo ChromaDB
EXTERNAL_GPT4O_API_KEY=   # chave de acesso ao endpoint iaedu.pt
```

> **Importante:** o ficheiro `.env` encontra-se listado no `.gitignore` e não deve ser incluído no repositório em nenhuma circunstância.

> **Palavras-passe com caracteres especiais:** caso a palavra-passe do MongoDB contenha os caracteres `@`, `#` ou `!`, deverá ser codificada previamente com o seguinte comando: `python3 -c "from urllib.parse import quote_plus; print(quote_plus('a_palavra_passe'))"`.

---

## Iniciar o servidor

```bash
cd PROJETO25-26_AI2/rag-api
uvicorn rag_api.api.main:app --reload
```

O servidor ficará disponível em `http://127.0.0.1:8000`.

A documentação interactiva dos endpoints (Swagger UI) encontra-se em `http://127.0.0.1:8000/docs`.

O registo de arranque confirma o estado do sistema:

```
{"event": "startup_complete", "rag_backend": "RAGController"}     ← pipeline ligado
{"event": "startup_complete", "rag_backend": "StubRAGController"}  ← pipeline indisponível
```

---

## Descrição dos endpoints

### Acesso público (sem autenticação)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/v1/health` | Verifica a disponibilidade do servidor |

### Utilizador autenticado (`rag:query`)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/v1/query` | Submete uma questão ao sistema RAG |
| POST | `/v1/feedback` | Regista uma avaliação sobre uma resposta obtida |

### Administrador (`rag:admin`)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/v1/applications` | Lista todas as aplicações registadas |
| POST | `/v1/applications` | Regista uma nova aplicação |
| GET | `/v1/applications/{id}` | Obtém os dados de uma aplicação |
| PATCH | `/v1/applications/{id}` | Actualiza os dados de uma aplicação |
| DELETE | `/v1/applications/{id}` | Remove uma aplicação |
| POST | `/v1/applications/{id}/keys` | Gera uma chave de acesso para uma aplicação |
| POST | `/v1/applications/{id}/keys/{hint}/rotate` | Efectua a rotação de uma chave de acesso |
| DELETE | `/v1/applications/{id}/keys/{hint}` | Revoga uma chave de acesso |
| POST | `/v1/embed` | Executa o pipeline de segmentação e indexação de documentos |

---

## Mecanismo de autenticação

Todos os pedidos autenticados requerem o seguinte cabeçalho HTTP:

```
Authorization: Bearer rag_xxxxxxxxxxxx
```

O sistema implementa controlo de acesso baseado em âmbitos de permissão (*scopes*):

| Âmbito | Permissões associadas |
|--------|----------------------|
| `rag:query` | Submeter questões ao RAG e registar avaliações de respostas |
| `rag:admin` | Gerir aplicações, chaves de acesso e executar o pipeline de indexação |

---

## Gestão de Aplicações e Chaves de Acesso

### Conceito

Uma **Aplicação** (*Application*) representa um cliente registado no sistema — por exemplo, o Portal Académico ou um serviço interno da instituição. Cada aplicação possui um conjunto de âmbitos de permissão e um limite máximo de pedidos por minuto.

Uma **Chave de Acesso** (*API Key*) é a credencial de autenticação da aplicação. Uma aplicação pode ter várias chaves activas em simultâneo, o que permite a rotação de credenciais sem interrupção do serviço.

```
Administrador
    │
    ├── regista a Aplicação "Portal Académico"
    ├── gera uma Chave de Acesso para essa aplicação
    └── partilha a chave com o utilizador responsável
                │
                └── utilizador emprega a chave para submeter questões ao RAG
```

### Procedimento completo

**1. Registar uma Aplicação**

```bash
curl -X POST http://127.0.0.1:8000/v1/applications \
  -H "Authorization: Bearer CHAVE_ADMIN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Portal Académico",
    "scopes": ["rag:query"],
    "rate_limit": 100
  }'
```

Resposta:
```json
{
  "id": "671d90c8-7183-424d-aa34-cbf31b20bfaa",
  "name": "Portal Académico",
  "scopes": ["rag:query"],
  "rate_limit": 100,
  "active": true
}
```

Conserve o campo `id` — é necessário para as operações seguintes.

**2. Gerar uma Chave de Acesso**

```bash
curl -X POST http://127.0.0.1:8000/v1/applications/671d90c8-.../keys \
  -H "Authorization: Bearer CHAVE_ADMIN"
```

Resposta:
```json
{
  "key": "rag_xK9mR...",
  "hint": "rag_xK9mR",
  "application_id": "671d90c8-...",
  "scopes": ["rag:query"]
}
```

> **Atenção:** o campo `key` é apresentado **uma única vez** no momento da criação. Deverá ser copiado imediatamente — não é possível recuperá-lo posteriormente. Em caso de perda, deverá ser gerada uma nova chave e a anterior revogada.

**3. Submeter uma questão ao RAG**

```bash
curl -X POST http://127.0.0.1:8000/v1/query \
  -H "Authorization: Bearer rag_xK9mR..." \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o prazo de matrícula?"}'
```

Resposta:
```json
{
  "answer": "O prazo de matrícula...",
  "sources": ["Regulamento de Matrículas, Artigo 5.º"],
  "trace_id": "ec919938-...",
  "session_id": "9eb6c5c4-..."
}
```

**4. Efectuar a rotação de uma chave (sem interrupção de serviço)**

```bash
curl -X POST http://127.0.0.1:8000/v1/applications/671d90c8-.../keys/rag_xK9mR/rotate \
  -H "Authorization: Bearer CHAVE_ADMIN"
```

A chave anterior é revogada e uma nova é devolvida de imediato. Ambas coexistem durante o período de transição, garantindo a continuidade do serviço.

**5. Revogar uma chave**

```bash
curl -X DELETE http://127.0.0.1:8000/v1/applications/671d90c8-.../keys/rag_xK9mR \
  -H "Authorization: Bearer CHAVE_ADMIN"
```

---

## Considerações de segurança

- As chaves de acesso são geradas com 256 bits de entropia criptográfica — nunca são armazenadas em claro, apenas o respectivo resumo criptográfico (SHA-256).
- Cada pedido é validado antes de qualquer execução — autenticação, âmbito de permissão e limite de taxa.
- O cabeçalho `Authorization` nunca é registado nos ficheiros de log.
- A revogação de uma chave produz efeito imediato, incluindo a invalidação da entrada em cache Redis.
- Em caso de comprometimento de uma chave, deverá ser revogada e substituída — o serviço não é interrompido durante este processo.

---

## Inicialização — criação do primeiro utilizador administrador

O primeiro utilizador administrador deverá ser criado directamente na base de dados MongoDB Atlas, uma vez que não existe endpoint público para este efeito por razões de segurança.

**Passo 1 — Gerar o par chave/resumo criptográfico**

```bash
python3 -c "
import secrets, hashlib
key = 'rag_' + secrets.token_urlsafe(32)
hash = hashlib.sha256(key.encode()).hexdigest()
hint = key[:12]
print('CHAVE: ', key)
print('HASH:  ', hash)
print('HINT:  ', hint)
"
```

**Passo 2 — Inserir o documento na colecção `api_keys` do Atlas**

```json
{
  "_id": "admin-bootstrap",
  "client_id": "admin-bootstrap",
  "key_hash": "<HASH gerado no passo anterior>",
  "key_hint": "<HINT gerado no passo anterior>",
  "scopes": ["rag:admin", "rag:query"],
  "rate_limit": 1000,
  "active": true,
  "created_at": { "$date": "2026-01-01T00:00:00Z" }
}
```

**Passo 3 — Utilizar a CHAVE gerada no Passo 1 em todos os pedidos de administração.**

---

## Resolução de problemas frequentes

| Erro | Causa provável | Resolução |
|------|----------------|-----------|
| `bad auth: authentication failed` | Palavra-passe do MongoDB incorrecta ou com caracteres especiais não codificados | Codificar a palavra-passe com `quote_plus` |
| `Connection refused` (Redis) | Serviço Redis não está em execução | `brew services start redis` |
| `invalid_key` (401) | Chave incorrecta ou resumo criptográfico errado no Atlas | Verificar o hash com `hashlib.sha256` |
| `insufficient_scope` (403) | Chave sem permissão para o endpoint solicitado | Verificar os âmbitos da Aplicação |
| `CHROMA_API_KEY required` | Variável de ambiente não definida no ficheiro `.env` | Solicitar a chave ao responsável pelo ChromaDB |
| `Model not found` (HuggingFace) | Modelo de embedding não se encontra em cache local | `python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` |
