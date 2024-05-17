import os
import typing

from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from dotenv import load_dotenv

CODER_SYSTEM_MESSAGE = """
Act as a Python developer.
You will receive an idea for what to code, and there will be a code reviewer 
who will review your code, so please repeat the idea in the code as comments, 
so that the reviewer can understand the context.
"""
REVIEWER_SYSTEM_MESSAGE = """
Act as a code reviewer for a Python developer.
You will review code written by the developer, and provide feedback.
When you feel it's good enough, you can explicitly echo "Good job!" to end the conversation.
"""


def load_llm_config() -> typing.Dict[str, str]:
    load_dotenv()
    llm_config = {
        "model": "gpt-4",
        "api_key": os.environ["AZURE_OPENAI_API_KEY"],
        "base_url": os.environ["AZURE_OPENAI_ENDPOINT"],
        "api_type": "azure",
        "api_version": "2023-06-01-preview",
    }
    return llm_config


def initiate_chat(prompt: str):
    llm_config = load_llm_config()
    initializer = UserProxyAgent(
        name="Init",
        code_execution_config=False,
    )
    coder = ConversableAgent(
        "Coder",
        system_message=CODER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    reviewer = ConversableAgent(
        "Reviewer",
        system_message=REVIEWER_SYSTEM_MESSAGE,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    def state_transition(last_speaker: Agent, groupchat: GroupChat):
        messages = groupchat.messages

        if last_speaker is initializer:
            return coder
        elif last_speaker is coder:
            return reviewer
        else:
            if "Good job!" in messages[-1]["content"]:
                return initializer
            else:
                return coder

    groupchat = GroupChat(
        agents=[initializer, coder, reviewer],
        messages=[],
        max_round=20,
        speaker_selection_method=state_transition,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    initializer.initiate_chat(manager, message=prompt)


def main():
    prompt = input("What's in your mind?: ")
    initiate_chat(prompt)


if __name__ == "__main__":
    main()
