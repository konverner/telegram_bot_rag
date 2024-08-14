from omegaconf import OmegaConf
from src.telegram_bot_rag.service.llm import FireworksLLM

config = OmegaConf.load("./tests/unit/conf/config.yaml")


def test_run():
	model_name = config.llm.model_name
	prompt_template = config.llm.prompt_template
	llm = FireworksLLM(model_name, prompt_template)
	query = "What is the capital of France?"
	document_name = "France"
	document_text = "The capital of France is Paris."

	actual_prompt = llm.prompt_template.format(
		query=query,
		document_name=document_name,
		document_text=document_text

	)

	assert document_name in actual_prompt
	assert document_text in actual_prompt
	assert query in actual_prompt
