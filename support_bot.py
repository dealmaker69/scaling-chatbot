# support_bot.py - Improved RAG Support Bot for clients (2026-ready)
import chainlit as cl
from dotenv import load_dotenv
import os
import pathlib

# ── LangChain imports ──
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── CONFIGURATION ── Customize per client
COMPANY_NAME = "Zorbas Oroklini"                        # Change this for each client
KNOWLEDGE_FOLDER = "Knowledge_Example"                  # Folder name (case-insensitive on Windows)
CHROMA_PERSIST_DIR = "./chroma_db_zorbas"               # Unique per client to avoid overlap
SHOW_SOURCES = True                                     # Set to False to hide sources

# LLM & Embeddings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings()

@cl.on_chat_start
async def setup_rag():
    msg = cl.Message(content=f"Loading {COMPANY_NAME} knowledge... please wait ⏳")
    await msg.send()

    vectorstore = None  # ← initialize to None so it's always defined

    try:
        # Make path robust
        knowledge_path = pathlib.Path(KNOWLEDGE_FOLDER).resolve()

        if not knowledge_path.exists():
            msg.content = f"Knowledge folder '{KNOWLEDGE_FOLDER}' not found. Please create it and add files."
            await msg.update()
            return

        # Try to load existing DB first (fast path)
        if os.path.exists(CHROMA_PERSIST_DIR):
            msg.content = "Loading existing knowledge base... ⏳"
            await msg.update()
            vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embeddings
            )
        else:
            msg.content = "Processing documents for the first time... this may take a minute ⏳"
            await msg.update()

            loader = DirectoryLoader(
                str(knowledge_path),
                glob="**/*",
                show_progress=True
            )
            docs = loader.load()

            if not docs:
                msg.content = "No readable documents found in the knowledge folder."
                await msg.update()
                return

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )

        # Now it's safe to use vectorstore — it must exist here
        if vectorstore is None:
            msg.content = "Failed to create or load vector store."
            await msg.update()
            return

        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # Your system prompt (with intent guidance added)
        system_prompt = f"""
You are a warm, professional and helpful customer support agent for {COMPANY_NAME}.
Always respond in a friendly tone.
Answer **only** using the provided context from official documents.
Be concise, clear and polite.

When the user asks about menu items, food availability, prices or ingredients — prioritize Availability documents.
When the user asks about closing time, opening hours, "what time do you close/open", or if something is open/closed tonight — **always** prioritize and use documents named containing "hours", "schedule", "closing", or "opening".



If the answer is not clearly in the context, or if the question requires human judgment (e.g. bookings, complaints, special requests), reply **exactly** with:

"I don't have that information right now — let me escalate this to a team member for you. One moment please! 🙌"

Do NOT guess, invent facts, or use knowledge outside the provided context.

Context:
{{context}}

User question:
{{input}}
        """

        prompt = ChatPromptTemplate.from_template(system_prompt)

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        cl.user_session.set("rag_chain", rag_chain)

        msg.content = f"✅ {COMPANY_NAME} support ready! How can I assist you today? 😊"
        await msg.update()

    except Exception as e:
        msg.content = f"Oops, error loading knowledge: {str(e)}"
        await msg.update()
@cl.on_message
async def main(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")

    if not rag_chain:
        await cl.Message(content="Support bot is still initializing — please wait a moment.").send()
        return

    try:
        # Show typing indicator
        await cl.Message(content="Typing...").send()

        result = await rag_chain.ainvoke({"input": message.content})
        answer = result["answer"].strip()

        source_text = ""
        if SHOW_SOURCES and "context" in result:
            sources = result["context"]
            if sources:
                files = set(doc.metadata.get("source", "unknown file") for doc in sources)
                source_text = "\n\n**Used sources:**\n" + "\n".join(f"• {os.path.basename(f)}" for f in files)

        full_response = answer + source_text

        await cl.Message(content=full_response).send()

    except Exception as e:
        await cl.Message(
            content=f"Sorry, I encountered an issue: {str(e)}\n\nPlease try rephrasing or start a new chat."
        ).send()