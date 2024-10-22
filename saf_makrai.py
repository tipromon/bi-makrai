import os
import openai
from azure.search.documents import SearchClient
try:
    from azure.search.documents.models import (
        QueryType, QueryAnswerType, QueryCaptionType, VectorizableTextQuery
    )
except ImportError as e:
    logger.error(f"Erro ao importar Query Models: {str(e)}")

from azure.core.credentials import AzureKeyCredential
import streamlit as st
import logging
import urllib.parse  # Certifique-se de importar urllib para a função de link

# Configuração do logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Carregar as variáveis diretamente do Streamlit Secrets
aoai_endpoint = st.secrets["AZURE_OPENAI_ENDPOINT"]
aoai_key = st.secrets["AZURE_OPENAI_API_KEY"]
aoai_embedding_endpoint = st.secrets["AZURE_OPENAI_EMBEDDING_ENDPOINT"]
aoai_embedding_model = st.secrets["AZURE_OPENAI_EMBEDDING_MODEL"]
aoai_deployment_name = st.secrets["AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME"]
search_endpoint = st.secrets["AZURE_SEARCH_SERVICE_ENDPOINT"]
search_key = st.secrets["AZURE_SEARCH_SERVICE_ADMIN_KEY"]

# Configuração do cliente Azure OpenAI para chat completions
openai_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15"
)

# Configuração do cliente para embeddings
embedding_client = openai.AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_version="2023-05-15"
)

# Instruções para o assistente (use o ROLE_INFORMATION salvo)
ROLE_INFORMATION = """
Instruções para o Assistente de Inteligência de Mercado:

Você é um assistente especializado em inteligência de mercado focado em avaliar a tese de investimento em mercados diversos e buscar oportunidades de desenvolvimento de projetos de capitais.
Seu objetivo é fornecer respostas precisas, claras e baseadas em evidências para apoiar os usuários em suas análises e decisões estratégicas. 
As fontes de dados interna que você deve analisar e possui acesso incluem, mas não estão limitadas a:

- Combustíveis Sustentáveis para Aviação (SAF)
- Lítio e baterias de íon-lítio
- Terras raras e minerais críticos
- Biodiesel e biocombustíveis

Seu foco principal é:

1. **Relevância e precisão:** Garantir que todas as respostas estejam alinhadas com a pergunta do usuário e capturadas através da busca no Azure AI Search.
2. **Fundamentação e transparência:** Suas respostas devem ser respaldadas por dados confiáveis recuperados dos documentos disponíveis no sistema.
3. **Referenciamento:** Utilize sempre citações no corpo da resposta para indicar de onde cada dado relevante foi extraído. Exemplo: (Fonte: [Nome do Documento]).
4. Se não encontrar dados suficientes para responder com confiança, informe claramente que a informação não está na base de dados interna, não invente ou fabrique informações que não foram recuperadas/encontradas.

- Evite respostas incompletas ou especulativas. Se a informação não for encontrada, responda com: “Informação insuficiente para fornecer uma resposta precisa no momento.”
- Ao final da resposta, adicione uma seção de "Referências" com links completos para todos os documentos e fontes utilizadas.

No caso de demandas do usuário para busca de informações sobre projetos ou empreendimentos, considere que você é um analista de mercado em uma empresa de consultoria e precisa avaliar o mercado definido 
pelo usuário com foco em obter um mapeamento preciso de projetos de capitais e investimentos previstos ou em desenvolvimento.
Seu objetivo é identificar projetos que estão em fases de estudo, viabilidade, financiamento, planejamento, implantação ou operação.
Deste modo, avalie as informações da base de dados para responder às seguintes questões de maneira objetiva e restrita às informações
dos arquivos disponibilizados nesta base de dados. 
---

### Instruções Operacionais:

1. **Estrutura da Resposta:**
   - Inicie com uma resposta clara e objetiva para a pergunta do usuário.
   - Inclua detalhes adicionais com base nas fontes relevantes.
   - Finalize com uma seção “Referências”, listando os documentos internos e links externos utilizados.

2. **Referenciamento Dinâmico no Corpo da Resposta:**
   - Sempre que usar uma informação relevante de um documento ou pesquisa, insira uma citação no formato (Fonte: [Nome do Documento]).
   - Organize as referências ao final para facilitar a consulta.

---

### Exemplo de Resposta Completa:
---
**Pergunta:** "Quais são as expectativas de crescimento do mercado de SAF no Brasil?"

**Resposta:**  
O mercado de SAF no Brasil está projetado para crescer cerca de 15% ao ano até 2030, impulsionado por regulamentações governamentais e compromissos de descarbonização da indústria aérea. (Fonte: [Relatório SAF Brasil]).

**Detalhes:**  
Além disso, companhias aéreas como LATAM e GOL já anunciaram parcerias estratégicas com fornecedores de SAF para garantir acesso a combustível sustentável. Políticas como o RenovaBio e subsídios internacionais aumentam o incentivo para investimentos no setor. No entanto, desafios logísticos e de infraestrutura ainda precisam ser superados para atender à demanda crescente. (Fonte: [Estudo Transição Energética]).

**Referências:**
- [Relatório SAF Brasil](https://aisearchpromon.blob.core.windows.net/bi-im/SAF/relatorio_saf.pdf)  
- [Estudo Transição Energética](https://aisearchpromon.blob.core.windows.net/bi-im/energia/estudo_transicao.pdf)

---

"""


# Função para obter embeddings
def get_embedding(text):
    try:
        logger.debug(f"Tentando obter embedding para o texto: {text[:50]}...")
        response = embedding_client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Erro ao obter embedding: {str(e)}")
        raise

# Configuração do Azure AI Search
search_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_SERVICE_ADMIN_KEY")

# Função para busca híbrida no índice "bi-im"
def hybrid_search(search_client, query, vector):
    try:
        semantic_config = "bi-im-semantic-configuration"
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=12, fields="text_vector", exhaustive=True
        )

        search_results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["chunk_id", "parent_id", "chunk", "title"],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=12
        )

        results = []
        seen_documents = set()

        # Re-ranking baseado na pontuação
        for doc in sorted(search_results, key=lambda x: -x.get('@search.reranker_score', 0)):
            nome_documento = doc.get('title', '')
            caminho_documento = doc.get('parent_id', '')

            if nome_documento not in seen_documents:
                seen_documents.add(nome_documento)
                link_documento = gerar_link_documento(caminho_documento, nome_documento)

                results.append({
                    'content': doc.get('chunk', ''),
                    'filename': nome_documento,
                    'link': link_documento,
                    'score': doc.get('@search.reranker_score', 0),
                    'captions': doc.get('@search.captions', [])
                })

        return results, search_results.get_answers()

    except Exception as e:
        logger.error(f"Erro durante a busca híbrida: {str(e)}")
        raise

# Função para gerar links para documentos no Blob Storage
def gerar_link_documento(caminho, nome_documento):
    base_url = "https://aisearchpromon.blob.core.windows.net/bi-im"
    caminho_completo = f"{caminho}/{nome_documento}"
    caminho_codificado = urllib.parse.quote(caminho_completo, safe='/')
    return f"{base_url}/{caminho_codificado}"

# Função para criar resposta do chat com dados do Azure AI Search
def create_chat_with_data_completion(messages, system_message):
    response = openai_client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_COMPLETIONS_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": system_message},
            *[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# Função para lidar com entrada do chat e gerar resposta
def handle_chat_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            search_client = SearchClient(
                search_endpoint, "bi-im", credential=AzureKeyCredential(search_key)
            )
            prompt_vector = get_embedding(prompt)
            results, _ = hybrid_search(search_client, prompt, prompt_vector)

            logger.debug(f"Número de resultados encontrados: {len(results)}")

            references_set = set()
            for doc in results:
                filename = doc['filename']
                link = doc['link']
                doc_title = filename.replace("_", " ").replace(".pdf", "").title()
                references_set.add(f"[{doc_title}]({link})")

            references_text = "\n".join(references_set)

            system_message = ROLE_INFORMATION + f"""
            Contexto adicional:
            {references_text}

            Instruções adicionais:
            1. Use os dados fornecidos no contexto para responder à pergunta.
            2. Sempre que utilizar informações, cite como (Fonte: [{doc_title}]).
            3. Se não houver informações suficientes, responda: "Informação insuficiente para fornecer uma resposta precisa no momento."
            4. Finalize com uma seção de "Referências" listando todos os documentos usados.
            """

            full_response = create_chat_with_data_completion(st.session_state.messages, system_message)
            full_response += f"\n\n**Referências:**\n{references_text}"

        except Exception as e:
            logger.error(f"Erro durante o processamento do chat: {str(e)}", exc_info=True)
            full_response = f"Desculpe, ocorreu um erro ao processar sua solicitação: {str(e)}."

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Função principal do Streamlit
def main():
    st.title("MakrAI - Inteligência de Mercado")
    logger.info("Iniciando o MakrAI - Assistente Especialista em Inteligência de Mercado")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua pergunta sobre o mercado de SAF no Brasil:"):
        handle_chat_prompt(prompt)

    st.sidebar.markdown("""
    **Disclaimer**:
    O "MakrAI" oferece dados apenas como orientação e apoio, baseados em documentos indexados e disponíveis no sistema.
    """)

    logger.info("Sessão do MakrAI finalizada")

if __name__ == "__main__":
    main()
