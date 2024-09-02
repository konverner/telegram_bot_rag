import logging.config
import os
from io import BytesIO

import telebot
from dotenv import find_dotenv, load_dotenv
from fastapi import UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from omegaconf import OmegaConf

from telegram_bot_rag.db.database import add_user, log_message
from telegram_bot_rag.service.exceptions import UnsupportedFileTypeException
from telegram_bot_rag.service.file_parser import FileParser
from telegram_bot_rag.service.llm import FireworksLLM
from telegram_bot_rag.service.vector_store import VectorStore

load_dotenv(find_dotenv(usecwd=True))  # Load environment variables from .env file

logging_config = OmegaConf.to_container(OmegaConf.load("./src/telegram_bot_rag/conf/logging_config.yaml"), resolve=True)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(usecwd=True))  # Load environment variables from .env file
TOKEN = os.getenv("BOT_TOKEN")

if TOKEN is None:
    logger.error("BOT_TOKEN is not set in the environment variables.")
    exit(1)

cfg = OmegaConf.load("./src/telegram_bot_rag/conf/config.yaml")
bot = telebot.TeleBot(TOKEN, parse_mode=None)

llm = FireworksLLM(model_name=cfg.llm.model_name, prompt_template=cfg.llm.prompt_template)
vector_store = VectorStore(embedding_model_name=cfg.retriever.model_name)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)

@bot.message_handler(commands=['start', 'help'])
def start(message):
    bot.send_message(message.chat.id, "Загрузите документы в формaте txt, docx, pdf и задавайте вопросы.")


@bot.message_handler(content_types=['document'])
def load_document(message):
    document = message.document
    logger.info(
        f"[load_document] Received document: {document.file_name} with type"
        f"{document.mime_type} from chat {message.from_user.username} ({message.chat.id})"
    )

    file_info = bot.get_file(document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_parser = FileParser(max_file_size_mb=10, allowed_file_types={"txt", "doc", "docx", "pdf"})

    try:
        upload_file = UploadFile(
            filename=document.file_name,
            file=BytesIO(downloaded_file),
            size=len(downloaded_file),
            headers={"content-type": document.mime_type}
        )
        document_text = file_parser.extract_content(upload_file)
        document_chunks = text_splitter.split_text(document_text)
        vector_store.upsert_documents(
            documents_text=document_chunks,
            documents_name=[document.file_name]*len(document_chunks),
            collection_name=message.from_user.username
        )

        logger.info(f"Document {document.file_name} has been upserted to ChromaDB")
        bot.send_message(message.chat.id, "Документ загружен.")
    except UnsupportedFileTypeException:
        logger.error(f"Document {document.file_name} has NOT been upserted to ChromaDB")
        bot.send_message(message.chat.id, "Пожалуйста, загрузите текстовый файл (txt, doc, docx, pdf).")


@bot.message_handler(commands=['get_docs'])
def get_docs(message):
    logger.info(f"[get_docs] Received message: '{message.text}' from chat {message.from_user.username} ({message.chat.id})")
    documents = vector_store.get_document_names(message.from_user.username)
    if not documents:
        response = "У вас нет загруженных документов."
        bot.send_message(message.chat.id, response)
    else:
        documents_str = '\n'.join(documents)
        response = f"Список ваших документов:\n {documents_str}"
        bot.send_message(message.chat.id, response)


@bot.message_handler(func=lambda message: True, content_types=['text'])
def ask_question(message):
    question = message.text
    logger.info(
        f"[ask_question] Received message: '{message.text}'"
        f"from chat {message.from_user.username} ({message.chat.id})"
    )
    retriever_results = vector_store.query(question, 1, message.from_user.username)
    document_text = retriever_results["documents"][0]
    document_name = retriever_results["metadatas"][0]
    response = llm.run(question, document_text, document_name)
    bot.send_message(message.chat.id, response)

    log_message(message.chat.id, message.text)
    add_user(
        message.chat.id, message.from_user.first_name,
        message.from_user.last_name, message.from_user.username,
        message.contact.phone_number if message.contact else None
    )


def start_bot():
    logger.info(f"bot `{str(bot.get_me().username)}` has started")
    bot.polling()
