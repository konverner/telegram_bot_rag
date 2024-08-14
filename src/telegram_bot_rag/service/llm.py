""" Application that provides functionality for the Telegram bot. """
import logging.config
import os

from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

load_dotenv(find_dotenv(usecwd=True))  # Load environment variables from .env file

# Load logging configuration with OmegaConf
logging_config = OmegaConf.to_container(OmegaConf.load("./src/telegram_bot_rag/conf/logging_config.yaml"), resolve=True)

# Apply the logging configuration
logging.config.dictConfig(logging_config)

# Configure logging
logger = logging.getLogger(__name__)

class FireworksLLM:
    def __init__(self, model_name: str, prompt_template: str):
        import fireworks.client
        API_KEY = os.getenv("API_KEY")
        if API_KEY is None:
            logger.error("API_KEY is not set in the environment variables.")
            raise ValueError("API_KEY is not set in the environment variables.")
        self.client = fireworks.client
        self.model_name = model_name
        fireworks.client.api_key = API_KEY
        self.prompt_template = prompt_template

    def run(self, query: str, document_text: list[str], document_name: list[str]):
        """Run the LLM model with the given query and document."""
        completion = self.client.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt_template.format(
                        query=query,
                        document_name=document_name[0],
                        document_text=document_text[0]
                    )
                }
            ],
            max_tokens=200,
            temperature=0.6,
            presence_penalty=0,
            frequency_penalty=0,
            top_p=1,
            top_k=40
        )
        return completion.choices[0].message.content
