import os
import typing

from autogen import Agent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from dotenv import load_dotenv


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
        system_message="Act as a Python developer.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    reviewer = ConversableAgent(
        "Reviewer",
        system_message="Act as a code reviewer for a Python developer.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    def state_transition(last_speaker: Agent, groupchat: GroupChat):
        if last_speaker is initializer:
            return coder
        elif last_speaker is coder:
            return reviewer
        else:
            return initializer

    groupchat = GroupChat(
        agents=[initializer, coder, reviewer],
        messages=[],
        max_round=20,
        speaker_selection_method=state_transition,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    initializer.initiate_chat(manager, message=prompt)


def main():
    prompt = input("Enter a prompt: ")
    initiate_chat(prompt)


if __name__ == "__main__":
    main()
