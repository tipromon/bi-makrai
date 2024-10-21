import os
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryType, QueryCaptionType, QueryAnswerType, VectorizableTextQuery
)
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

# Configuração dos clientes Azure OpenAI
openai_client = openai.AzureOpenAI(
    api_key=aoai_key,
    azure_endpoint=aoai_endpoint,
    api_version="2023-05-15"
)

embedding_client = openai.AzureOpenAI(
    api_key=aoai_key,
    azure_endpoint=aoai_embedding_endpoint,
    api_version="2023-05-15"
)

# Instruções detalhadas para o assistente especialista em SAF
ROLE_INFORMATION = """
Instruções para o Assistente de Inteligência Artificial Especializado em SAF:
...
"""

# Função para obter embeddings
def get_embedding(text):
    try:
        logger.debug(f"Tentando obter embedding para o texto: {text[:50]}...")
        response = embedding_client.embeddings.create(
            model=aoai_embedding_model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Erro ao obter embedding: {str(e)}")
        raise

# Função para realizar busca híbrida no índice "vector-saf"
def hybrid_search(search_client, query, vector):
    try:
        semantic_config = "vector-saf-semantic-configuration"
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=5, fields="text_vector", exhaustive=True
        )
        search_results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=["chunk_id", "parent_id", "chunk", "title"],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name=semantic_config,
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            top=5
        )

        results = []
        seen_documents = set()

        for doc in search_results:
            nome_documento = doc.get('title', '')
            if nome_documento not in seen_documents:
                seen_documents.add(nome_documento)
                link_documento = gerar_link_documento(nome_documento)
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

# Função para gerar o link do documento no Blob Storage
def gerar_link_documento(nome_documento):
    base_url = "https://aisearchpromon.blob.core.windows.net/bi-saf"
    nome_documento_codificado = urllib.parse.quote(nome_documento)
    return f"{base_url}/{nome_documento_codificado}"

# Função para lidar com a entrada do chat e gerar resposta
def handle_chat_prompt(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            search_client = SearchClient(
                search_endpoint, "vector-saf", credential=AzureKeyCredential(search_key)
            )
            prompt_vector = get_embedding(prompt)
            results, semantic_answers = hybrid_search(search_client, prompt, prompt_vector)

            logger.debug(f"Número de resultados encontrados: {len(results)}")

            references_set = set()
            for doc in results:
                filename = doc['filename']
                link = doc['link']
                references_set.add(f"[{filename}]({link})")

            references_text = "\n".join(references_set)

            system_message = ROLE_INFORMATION + f"""
            Contexto adicional:
            {references_text}

            Instruções adicionais:
            1. Use as informações fornecidas no contexto acima para responder à pergunta do usuário.
            2. Cite as fontes relevantes usando o formato [nome do documento].
            3. Se não houver informações suficientes, diga que não sabe.
            4. Ao final da resposta, adicione uma seção de "Referências" com os links para os documentos utilizados.
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
    st.title("MakrAI - Especialista em SAF")
    logger.info("Iniciando o MakrAI - Assistente Especialista em SAF")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua pergunta sobre o mercado de SAF no Brasil:"):
        handle_chat_prompt(prompt)

    st.sidebar.markdown("""
    **Disclaimer**:
    O "MakrAI" tem como único objetivo disponibilizar dados que sirvam como um meio de orientação e apoio. As informações fornecidas são baseadas em documentos indexados e disponíveis no sistema.
    """)

    logger.info("Sessão do MakrAI finalizada")

if __name__ == "__main__":
    main()
