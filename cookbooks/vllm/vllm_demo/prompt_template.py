# Copyright 2025 Fujitsu Research of America, Inc.
#
# This software is licensed under an End User License Agreement (EULA) by Fujitsu
# Research of America, Inc. You are not allowed to use, copy, modify, or distribute
# this software and its documentation without express permission from Fujitsu Research
# of America, Inc. Please refer to the full EULA provided with this software for
# detailed information on permitted uses and restrictions.
#
# The software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a
# particular purpose and noninfringement. In no event shall Fujitsu Research of America,
# Inc. be liable for any claim, damages or other liability, whether in an action of
# contract, tort or otherwise, arising from, out of or in connection with the software
# or the use or other dealings in the software.


from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate


class PromptTemplates:
    """Manage all prompt templates"""

    @classmethod
    def get_qa_system_prompt(cls) -> ChatMessage:
        """Get QA system prompt"""
        return ChatMessage(
            content=(
                "You are an expert Q&A system that is trusted around the world.\n"
                "Always answer the query using the provided context information, and not prior knowledge.\n"
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."
                # "\n3. Answer concisely in 50 words or less."
            ),
            role=MessageRole.SYSTEM,
        )

    @classmethod
    def get_qa_prompt_messages(cls) -> list:
        """Get QA prompt messages"""
        return [
            cls.get_qa_system_prompt(),
            ChatMessage(
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
                role=MessageRole.USER,
            ),
        ]

    @classmethod
    def get_chat_qa_prompt(cls) -> ChatPromptTemplate:
        """Get chat QA prompt template"""
        return ChatPromptTemplate(message_templates=cls.get_qa_prompt_messages())

    @classmethod
    def get_refine_prompt_messages(cls) -> list:
        """Get refine prompt messages"""
        return [
            cls.get_qa_system_prompt(),
            ChatMessage(
                content=(
                    "You are an expert Q&A system that strictly operates in two modes when refining existing answers:\n"
                    "1. **Rewrite** an original answer using the new context.\n"
                    "2. **Repeat** the original answer if the new context isn't useful.\n"
                    "Never reference the original answer or context directly in your answer.\n"
                    "When in doubt, just repeat the original answer.\n"
                    "New Context: {context_msg}\n"
                    "Query: {query_str}\n"
                    "Original Answer: {existing_answer}\n"
                    "New Answer: "
                ),
                role=MessageRole.USER,
            ),
        ]

    @classmethod
    def get_chat_refine_prompt(cls) -> ChatPromptTemplate:
        """Get chat refine prompt template"""
        return ChatPromptTemplate(message_templates=cls.get_refine_prompt_messages())
