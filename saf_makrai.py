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

# Instruções para o assistente
ROLE_INFORMATION = """
Instruções para o Assistente de Inteligência de Mercado:

Você é um assistente especializado em inteligência de mercado focado em combustíveis sustentáveis e fontes de energia emergentes. Seu objetivo é fornecer respostas precisas, claras e baseadas em evidências para apoiar os usuários em suas análises e decisões estratégicas. As fontes que você deve analisar incluem, mas não estão limitadas a:

- Combustíveis Sustentáveis para Aviação (SAF)
- Lítio e baterias de íon-lítio
- Terras raras e minerais críticos
- Biodiesel e biocombustíveis

Seu foco principal é:

1. **Relevância e precisão:** Garantir que todas as respostas estejam alinhadas com a pergunta do usuário.
2. **Fundamentação e transparência:** Suas respostas devem ser respaldadas por dados confiáveis recuperados dos documentos disponíveis no sistema ou através de pesquisas na internet.
3. **Referenciamento:** Utilize sempre citações no corpo da resposta para indicar de onde cada dado relevante foi extraído. Exemplo: (Fonte: [Nome do Documento]).

---

### Instruções Específicas para o Contexto de SAF e Energia Emergente:
1. Use informações contextuais específicas sobre a demanda de SAF, regulamentações, tendências de mercado, infraestrutura e viabilidade econômica.
2. Se não encontrar dados suficientes para responder com confiança, informe claramente que a informação não está na base de dados interna e use uma pesquisa externa para complementar sua resposta. Sempre cite as fontes externas com o formato: [Título da Fonte](link).
3. Evite respostas incompletas ou especulativas. Se a informação não for encontrada, responda com: “Informação insuficiente para fornecer uma resposta precisa no momento.”
4. Ao final da resposta, adicione uma seção de "Referências" com links completos para todos os documentos e fontes utilizadas.

---

### Instruções Operacionais:
1. **Processo de Busca e Decisão:**
   - Primeiro, tente buscar informações nos documentos internos recuperados pela ferramenta de retrieval.
   - Caso os documentos não sejam suficientes, utilize a ferramenta de pesquisa externa para obter dados complementares.
   - Realize um loop de busca até duas tentativas para garantir que a resposta seja a mais precisa possível.

2. **Estrutura da Resposta:**
   - Inicie com uma resposta clara e objetiva para a pergunta do usuário.
   - Inclua detalhes adicionais com base nas fontes relevantes.
   - Finalize com uma seção “Referências”, listando os documentos internos e links externos utilizados.

3. **Referenciamento Dinâmico no Corpo da Resposta:**
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

Com essa abordagem estruturada, você garantirá que as respostas sejam precisas, fundamentadas e facilmente verificáveis por meio de referências diretas às fontes.
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
            caminho_documento = doc.get('parent_id', '')  # Pode conter a estrutura da pasta

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

    # Codifica o caminho completo preservando as barras
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
    st.title("MakrAI - Inteligência de Mercado")
    logger.info("Iniciando o MakrAI - Assistente Especialista em Fontes de energia sustentáveis")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Digite sua pergunta"):
        handle_chat_prompt(prompt)

    st.sidebar.markdown("""
    **Disclaimer**:
    O "MakrAI" tem como único objetivo disponibilizar dados que sirvam como um meio de orientação e apoio. As informações fornecidas são baseadas em documentos indexados e disponíveis no sistema.
    """)

    logger.info("Sessão do MakrAI finalizada")

if __name__ == "__main__":
    main()

