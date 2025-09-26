from enum import Enum


# SYSTEM_MESSAGE = """
# You are an expert in Maximal Extractable Value (MEV) that answers questions using the tools at your disposal.
# These tools have information regarding MEV research including academic papers, articles, diarized transcripts from conversations registered on talks at conferences or podcasts.
# Here are some guidelines that you must follow:
# * For any user message that is not related to MEV, blockchain, or mechanism design, respectfully decline to respond and suggest that the user ask a relevant question.
# * If your tools are unable to find an answer, you should say that you haven't found an answer.
#
# Now answer the following question:
# {question}
# """.strip()

#  that answers questions using the query tools at your disposal.

# If the user requested sources or content, return the sources regardless of response worded by the query tool.

# REACT_CHAT_SYSTEM_HEADER is the chat format used to determine the action e.g. if the query tool should be used or not.
# It is tweaked from the base one.

# You are designed to help with a variety of tasks, from answering questions \
# to providing summaries to providing references and sources about the requested content.


# You are responsible for using
# the tool in any sequence you deem appropriate to complete the task at hand.
# This may require breaking the task into subtasks and using different tools
# to complete each subtask.


LLM_TEMPERATURE = 0
NUMBER_OF_CHUNKS_TO_RETRIEVE = 10
TEXT_SPLITTER_CHUNK_SIZE = 700
TEXT_SPLITTER_CHUNK_OVERLAP_PERCENTAGE = 10

'''
valid OpenAI model name in: gpt-4, gpt-4-32k, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, 
gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, 
text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-3.5-turbo-0125
'''
OPENAI_INFERENCE_MODELS = ["gpt-3.5-turbo-0125", "gpt-4", "gpt-4-32k", "gpt-4-0613", "gpt-4-32k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613",
"gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0301", "text-davinci-003", "text-davinci-002", "gpt-3.5-turbo-instruct", "gpt-35-turbo-16k", "gpt-35-turbo", "gpt-4-1106-preview", "gpt-4-turbo", "gpt-4-turbo-1106"]


class DOCUMENT_TYPES(Enum):
    YOUTUBE_VIDEO = "youtube_video"
    RESEARCH_PAPER = "research_paper"
    ARTICLE = "article"

