import os
from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def feed_llm(query):

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatGroq(temperature=0, model_name=os.environ["MODEL_NAME"])

    vector_store = PineconeVectorStore(embedding=embedding, index_name=os.environ["INDEX_NAME"])

    template = """
        You are analyzing historical product demand data to make predictions. Based on the following daily demand records:

        {context}

        Question: {question}

        Return ONLY a numerical value representing the predicted demand. No explanation or additional text.
        """

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
            {"context": vector_store.as_retriever() | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
    )

    res = rag_chain.invoke(query)

    return res


