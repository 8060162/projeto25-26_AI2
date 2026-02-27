# projeto25-26_AI2

## ⚙️ Fluxo de Trabalho GitHub

### **Branches**
* **main**: Código estável e validado em estado de produção.
* **develop**: Branch de integração onde as novas funcionalidades são testadas em conjunto.
* **feature/issue-[ID]**: Desenvolvimento isolado de funcionalidades, criadas a partir da develop.

### **Commits (Conventional Commits)**
A estrutura deve ser concisa e informativa: `<tipo>(<âmbito>): <descrição curta>`

**Exemplo:**
`feat(indexer): implementar extração hierárquica com Unstructured`

**Significados:**
* **feat**: Introdução de nova funcionalidade.
* **fix**: Correção de um erro técnico.
* **docs**: Alterações na documentação.
* **refactor**: Melhoria no código que não altera o seu comportamento.

### **Merge & Pull Requests**
* As funcionalidades são fundidas na `develop` via Pull Request.
* A `main` é atualizada apenas quando a `develop` atinge um estado estável de Milestone.
* O merge deve incluir a referência para fechar a Issue correspondente (ex: `Closes #1`).

### **Labels**
* **milestone-1**: Tarefas pertencentes à fase inicial da pipeline.
* **retrieval**: Relacionado com a pesquisa e base de dados vetorial.
* **generation**: Relacionado com a integração da OpenAI e resposta final.
* **bug**: Identificação de comportamentos inesperados.

---

## 🚀 Tecnologias Principais
* **Python**
* **OpenAI API**
* **BGE-M3 Embeddings**
* **ChromaDB**
* **Unstructured / LangChain**

## 📋 Milestone 1: Fundação da Pipeline RAG

A Milestone 1 foca-se na construção da infraestrutura base para o sistema de **Geração Aumentada por Recuperação (RAG)**. O objetivo central é garantir que as respostas do LLM sejam fundamentadas em conhecimento externo estruturado, mitigando alucinações através de uma pipeline de dados rigorosa.

### **Issue #1: Extração e Estruturação de Dados**
* **Descrição:** Implementação do módulo de ingestão utilizando a biblioteca *Unstructured* para processamento de PDFs.
* **Objetivo:** Criar um dicionário hierárquico de 3 níveis (Título > Capítulo > Artigo) para preservar a integridade semântica.
* **Entrega:** Ficheiros intermédios (Markdown/JSON) para auditoria e validação da extração.

### **Issue #2: Vetorização e Armazenamento (Embeddings)**
* **Descrição:** Conversão de texto em representações vetoriais através do modelo **BGE-M3**.
* **Chunking Estrutural:** Utilização de *Recursive Chunking* com o **Artigo** como unidade mínima atómica, respeitando a janela de contexto do modelo.
* **Base de Dados:** Persistência no **ChromaDB**, permitindo a recuperação eficiente por proximidade matemática.

### **Issue #3: Motor de Recuperação (Retriever)**
* **Descrição:** Desenvolvimento da lógica de pesquisa para identificar os segmentos de texto mais relevantes.
* **Objetivo:** Implementar a procura por similaridade de cosseno para capturar a intenção semântica do utilizador.
* **Rastreabilidade:** Utilização de metadados para garantir que cada resposta possa ser mapeada de volta à sua fonte original.

### **Issue #4: Módulo de Geração (Generator) com OpenAI**
* **Descrição:** Integração da pipeline com modelos da **OpenAI** para a geração da resposta final.
* **Objetivo:** Configuração de *System Prompts* que forçam a fidelidade ao contexto injetado.
* **Citação Automática:** O modelo é instruído a citar explicitamente a fonte (ex: "Artigo X do Regulamento Y"), garantindo transparência e precisão.