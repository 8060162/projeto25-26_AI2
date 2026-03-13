import os
import json
import fitz  # PyMuPDF para extração rápida
from openai import OpenAI

class QwenExtremeParser:
    def __init__(self, api_key=None):
        # Tenta ler a chave do ambiente se não for passada
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # qwen-max é recomendado para reconstrução de tabelas e layouts de colunas
        self.model = "qwen-max"

    def _get_pdf_text(self, pdf_path):
        """Extrai o texto bruto com marcação de página."""
        text_blocks = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_blocks.append(f"[PÁGINA {page.number + 1}]\n{page.get_text()}")
        return "\n".join(text_blocks)

    def process_corrupted_document(self, pdf_path, doc_id_fallback):
        """
        Envia o documento para o Qwen reconstruir a estrutura legal.
        """
        raw_text = self._get_pdf_text(pdf_path)
        
        prompt = f"""
        CONTEXTO: Estás a processar um regulamento jurídico do Instituto Politécnico do Porto (P.PORTO).
        PROBLEMA: O texto extraído contém erros de codificação (mojibake) e o layout de colunas pode estar misturado.
        MISSÃO: Reconstrói o documento e extrai a estrutura exata para o formato JSON abaixo.

        REGRAS:
        1. Ignora índices (tabelas de conteúdo com ......), cabeçalhos e rodapés.
        2. Corrige o texto corrompido (ex: 'N]^‘V‡[^WU' -> 'Objetivos') usando lógica jurídica.
        3. Identifica corretamente os capítulos e agrupa os artigos dentro deles.
        4. Garante que o campo 'pagina' corresponde ao número da página real no PDF.

        ESTRUTURA JSON OBRIGATÓRIA:
        {{
          "document_info": {{ 
             "doc_id": "{doc_id_fallback}", 
             "ano": "Extrair do texto", 
             "status": "VIGENTE" 
          }},
          "preambulo": "Texto dos considerandos antes do Artigo 1",
          "estrutura": {{
            "CAP_1": {{
              "titulo": "NOME DO CAPÍTULO EM MAIÚSCULAS",
              "artigos": {{
                "ART_1": {{
                  "titulo": "TÍTULO DO ARTIGO EM MAIÚSCULAS",
                  "conteudo": "Texto completo e limpo",
                  "pagina": 0
                }}
              }}
            }}
          }}
        }}

        TEXTO BRUTO PARA PROCESSAR:
        {raw_text}
        """

        try:
            print(f"[*] A invocar inteligência do Qwen para backup extremo: {doc_id_fallback}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "És um parser de documentos JSON especializado em direito português. Responde apenas com JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0 # Rigidez total na resposta
            )
            
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"[ERRO CRÍTICO NO BACKUP]: {e}")
            return None

# --- Exemplo de ativação manual para teste ---
if __name__ == "__main__":
    # Exemplo de uso para o ficheiro 633 que estava corrompido
    backup_parser = QwenExtremeParser(api_key="TUA_API_KEY")
    path = "data/raw/Regulamento nº 633-2024_Alteração ao Regulamento de Avaliação de Aproveitamento.pdf"
    
    final_json = backup_parser.process_corrupted_document(path, "Regulamento_633_2024")
    
    if final_json:
        output_name = "data/processed/RECONSTRUIDO_633_2024.json"
        with open(output_name, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
        print(f"[SUCESSO] Documento reconstruído em: {output_name}")