# pylint: disable=invalid-name,line-too-long
"""
Adapted from
https://github.com/JGalego/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra_chat_anthropic.py
"""
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatAnthropic as Anthropic
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever

class bcolors:  #pylint: disable=too-few-public-methods
    """
    ANSI escape sequences
    https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

MAX_HISTORY_LENGTH = 5

def build_chain():
    """
    Builds the LangChain chain
    """
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
    kendra_index_id = os.environ["KENDRA_INDEX_ID"]

    llm = Anthropic(
        temperature=0,
        anthropic_api_key=anthropic_api_key,
        max_tokens_to_sample=512
    )

    retriever = AmazonKendraRetriever(index_id=kendra_index_id)

    prompt_template = """
Human: This is a friendly conversation between a human and an AI. 
The AI is talkative and provides specific details from its context but limits it to 240 tokens.
If the AI does not know the answer to a question, it truthfully says it does not know.

Assistant: OK, got it, I'll be a talkative truthful AI assistant.

Human: Here are a few documents in <documents> tags:
<documents>
{context}
</documents>
Based on the above documents, provide a detailed answer for, {question} Answer "don't know" 
if not present in the document. 

Assistant:"""

    qa_prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    condense_qa_template = """
Given the following conversation and a follow up question, rephrase the follow up question 
to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

    # Initialize QA chain with chat history
    # https://python.langchain.com/en/latest/modules/chains/index_examples/chat_vector_db.html
    qa = ConversationalRetrievalChain.from_llm(  #
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )

    return qa

def run_chain(chain, prompt: str, history=None):
    """
    Runs the Q&A chain given a user prompt and chat history
    """
    if history is None:
        history = []
    return chain({"question": prompt, "chat_history": history})

def prompt_user():
    """
    Helper function to get user input
    """
    print(f"{bcolors.OKBLUE}Hello! How can I help you?{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}Ask a question, start a New search: or Stop cell execution to exit.{bcolors.ENDC}")
    return input(">")

if __name__ == "__main__":
    # Initialize chat history
    chat_history = []

    # Initialize Q&A chain
    qa_chain = build_chain()

    try:
        while query := prompt_user():
            # Process user input in case of a new search
            if query.strip().lower().startswith("new search:"):
                query = query.strip().lower().replace("new search:", "")
                chat_history = []
            if len(chat_history) == MAX_HISTORY_LENGTH:
                chat_history.pop(0)

            # Show answer and keep a record
            result = run_chain(qa_chain, query, chat_history)
            chat_history.append((query, result["answer"]))
            print(f"{bcolors.OKGREEN}{result['answer']}{bcolors.ENDC}")

            # Show sources
            if 'source_documents' in result:
                print(bcolors.OKGREEN + 'Sources:')
                for doc in result['source_documents']:
                    print(f"+ {doc.metadata['source']}")
    except KeyboardInterrupt:
        pass
