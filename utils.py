import os
import pickle
import langchain

import faiss
from langchain import HuggingFaceHub, PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    StringPromptTemplate
)
from langchain.output_parsers import PydanticOutputParser
from langchain.tools.json.tool import JsonSpec

from typing import List, Union, Callable
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.text_splitter import CharacterTextSplitter
from custom_faiss import MyFAISS
from langchain.cache import InMemoryCache
from langchain.chat_models import ChatGooglePalm
from langchain.document_loaders import JSONLoader
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, BaseMultiActionAgent
from langchain.tools import StructuredTool
from langchain.chains import create_tagging_chain
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from pydantic import BaseModel, Field
from typing import Optional

class ToolArgsSchema(BaseModel):
    student_name: Optional[str] = Field(description="The name of the student")
    question: str = Field(description="The question being asked")
    question_type: str = Field(description="The type of question being asked")
    interest: Optional[str] = Field(description="The interest of the student")

    class Config:
        schema_extra = {
            "required": ["question", "question_type"]
        }





langchain.llm_cache = InMemoryCache()

model_name = "GPT-4"

pickle_file = "_vs.pkl"
index_file = "_vs.index"
models_folder = "models/"
os.environ["LANGCHAIN_TRACING"] = "true"
discussions_file_path = "discussion_entries.json"

llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, verbose=True)

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

chat_history = []

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10)

vectorstore_index = None

agent_prompt = """
I am the LLM AI canvas discussion grading assistant. 
I can answer two types of questions: grade-based questions and interest-based questions. 
Grade-based questions are about the grades of a certain student or a group of students based on the rubric below for the canvas discussion on the topic 8 nouns. 
Interest-based questions are about the interests or skills of a certain student or a group of students based on their discussion posts.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about type of question it is
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        print("llm_output")
        print(llm_output)
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

system_template = """
I am the LLM AI canvas discussion grading assistant. 
I can answer two types of questions: grade-based questions and interest-based questions. 
Grade-based questions are about the grades of a certain student or a group of students based on the rubric below for the canvas discussion on the topic 8 nouns. 
Interest-based questions are about the interests or skills of a certain student or a group of students based on their discussion posts.
To grade student discussions, I will follow the rubric below.

Student Post

3 points: Post includes 8 nouns and text describing how these nouns relate to the student.
2 points: Student's post includes 8 nouns but does not offer how those nouns relate to the student.
1 point: Student's post has significant missing details.
0 points: The student does not provide an initial post, or otherwise does not follow assignment instructions.


Response to Others

3 points: Student responds to at least 3 other student discussion threads AND responds to questions asked of them. Student posts insightful comments that prompt on target discussion. These posts also avoid throw away comments such as I agree, Me too, Good idea.
2 points: Student was notably lacking in one criterion.
1 point: Student was notably lacking in two criteria.
0 points: The student does not interact in the threads of other students.
I will be able to identify each student by name, and I will be able to share their likings, interests, and other characteristics. I will also be able to filter out students based on their interests.

I will not deviate from the grading scheme. I will grade each discussion entry and reply carefully, and I will share the grades of all individuals by name on the basis of the rubric with final score.

The discussions and their replies are in following format:
Student Post: Student Name
Reply to: Another Student Discussion ID

Following are the relevant discussions to grade or answer the interest based questions
----------------
Discussions: 
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


def set_model_and_embeddings():
    global chat_history
    # set_model(model)
    # set_embeddings(model)
    chat_history = []

def set_embeddings(model):
    global embeddings
    if model == "GPT-3.5" or model == "GPT-4":
        print("Loading OpenAI embeddings")
        embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    elif model == "Flan UL2" or model == "Flan T5":
        print("Loading Hugging Face embeddings")
        embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")


def get_search_index():
    global vectorstore_index, model_name
    if os.path.isfile(get_file_path(model_name, pickle_file)) and os.path.isfile(
            get_file_path(model_name, index_file)) and os.path.getsize(get_file_path(model_name, pickle_file)) > 0:
        # Load index from pickle file
        with open(get_file_path(model_name, pickle_file), "rb") as f:
            # search_index = Chroma(persist_directory=models_folder, embedding_function=embeddings)
            search_index = pickle.load(f)
            print("Loaded index")
    else:
        search_index = create_index(model_name)
        print("Created index")

    vectorstore_index = search_index
    return search_index


def create_index(model):
    source_chunks = create_chunk_documents()
    search_index = search_index_from_docs(source_chunks)
    # search_index.persist()
    faiss.write_index(search_index.index, get_file_path(model, index_file))
    # Save index to pickle file
    with open(get_file_path(model, pickle_file), "wb") as f:
        pickle.dump(search_index, f)
    return search_index


def get_file_path(model, file):
    # If model is GPT3.5 or GPT4 return models_folder + openai + file else return models_folder + hf + file
    if model == "GPT-3.5" or model == "GPT-4":
        return models_folder + "openai" + file
    else:
        return models_folder + "hf" + file


def search_index_from_docs(source_chunks):
    # print("source chunks: " + str(len(source_chunks)))
    # print("embeddings: " + str(embeddings))

    search_index = MyFAISS.from_documents(source_chunks, embeddings)
    return search_index


def get_html_files():
    loader = DirectoryLoader('docs', glob="**/*.html", loader_cls=UnstructuredHTMLLoader, recursive=True)
    document_list = loader.load()
    for document in document_list:
        document.metadata["name"] = document.metadata["source"].split("/")[-1].split(".")[0]
    return document_list

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["name"] = record.get("name")
    return metadata
def get_json_file():
    global discussions_file_path
    loader = JSONLoader(
        file_path=discussions_file_path,
        jq_schema='.[]', metadata_func=metadata_func, content_key="message")
    return loader.load()
def fetch_data_for_embeddings():
    # document_list = get_text_files()
    document_list = get_html_files()
    # document_list = get_json_file()
    print("document list: " + str(len(document_list)))
    return document_list


def get_text_files():
    loader = DirectoryLoader('docs', glob="**/*.txt", loader_cls=TextLoader, recursive=True)
    document_list = loader.load()
    return document_list


def create_chunk_documents():
    sources = fetch_data_for_embeddings()

    splitter = CharacterTextSplitter(separator=" ", chunk_size=800, chunk_overlap=0)

    source_chunks = splitter.split_documents(sources)

    print("chunks: " + str(len(source_chunks)))

    return sources


def get_qa_chain(vectorstore_index, question, metadata):
    global llm, model_name
    print(llm)
    filter_dict = {"name": metadata.student_name}
    # embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    # compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=gpt_3_5_index.as_retriever())
    retriever = get_retriever(filter_dict, vectorstore_index, metadata)

    print(retriever.get_relevant_documents(question))

    chain = ConversationalRetrievalChain.from_llm(llm, retriever, return_source_documents=True,
                                                  verbose=True, get_chat_history=get_chat_history,
                                                  combine_docs_chain_kwargs={"prompt": CHAT_PROMPT})
    return chain


def get_retriever(filter_dict, vectorstore_index, metadata):
    if metadata.question_type == "grade-based":
        retriever = vectorstore_index.as_retriever(search_type='mmr',
                                                   search_kwargs={'lambda_mult': 1, 'fetch_k': 20, 'k': 10,
                                                                  'filter': filter_dict})

    else:
        retriever = vectorstore_index.as_retriever(search_type='mmr',
                                                   search_kwargs={'lambda_mult': 1, 'fetch_k': 20, 'k': 10})

    return retriever


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


def generate_answer(question, metadata:  ToolArgsSchema) -> str:
    # print("filter: " + filter)
    global chat_history, vectorstore_index
    chain = get_qa_chain(vectorstore_index, question, metadata)

    result = chain(
        {"question": question, "chat_history": chat_history})
    chat_history.extend([(question, result["answer"])])
    sources = []
    print(result)

    for document in result['source_documents']:
        source = document.metadata['source']
        sources.append(source.split('/')[-1].split('.')[0])
        print(sources)

    source = ',\n'.join(set(sources))
    # return result['answer'] + '\nSOURCES: ' + source
    return result['answer']
def get_question_type(question):

    parser = PydanticOutputParser(pydantic_object=ToolArgsSchema)
    prompt_template = """I can answer two types of questions: grade-based questions and interest-based questions. 
Grade-based questions are about the grades of a certain student or a group of students based on the rubric below for the canvas discussion on the topic 8 nouns. 
Interest-based questions are about the interests or skills of a certain student or a group of students based on their discussion posts.
Question: {question}
Find following information about the question asked. Return Optional empty if the information is not available.:
Format instructions: {format_instructions}"""

    llm = OpenAI(temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"], output_parser=parser, partial_variables={"format_instructions": parser.get_format_instructions()})
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,

    )
    output = llm_chain.run(question)
    output = parser.parse(output)
    output = generate_answer(question, output)
    return output











# class FakeAgent(BaseMultiActionAgent):
#     """Fake Custom Agent."""
#
#     @property
#     def input_keys(self):
#         return ["input"]
#
#     def plan(
#             self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
#     ) -> Union[List[AgentAction], AgentFinish]:
#         print("input keys")
#         print(self.input_keys)
#         print("intermediate steps")
#         print(intermediate_steps)
#         print("kwargs")
#         print(kwargs)
#
#         """Given input, decided what to do.
#
#         Args:
#             intermediate_steps: Steps the LLM has taken to date,
#                 along with observations
#             **kwargs: User inputs.
#
#         Returns:
#             Action specifying what tool to use.
#         """
#         if len(intermediate_steps) == 0:
#             first_action = AgentAction(tool="question type", tool_input=kwargs["input"], log="")
#             print("first action")
#             print(first_action)
#             second_action = AgentAction(tool="Grade",tool_input=kwargs["input"], log="")
#             print("second action")
#             print(second_action)
#             return [
#                 first_action,
#                 second_action,
#             ]
#         else:
#             return AgentFinish(return_values={"output": "bar"}, log="")
#
#     async def aplan(
#             self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
#     ) -> Union[List[AgentAction], AgentFinish]:
#         """Given input, decided what to do.
#
#         Args:
#             intermediate_steps: Steps the LLM has taken to date,
#                 along with observations
#             **kwargs: User inputs.
#
#         Returns:
#             Action specifying what tool to use.
#         """
#         if len(intermediate_steps) == 0:
#             return [
#                 AgentAction(tool="question type", tool_input=kwargs["input"], log=""),
#                 AgentAction(tool="Grade",
#                             tool_input={
#                                 "student_name": kwargs["student_name"],
#                                 "question": kwargs["question"],
#                                 "question_type": kwargs["question_type"],
#                                 "interest": kwargs["interest"]
#                             }, log=""),
#             ]
#         else:
#             return AgentFinish(return_values={"output": "bar"}, log="")
#
#
# schema = {
#     "properties": {
#         "student_name" : {"type": "string", "description": "The name of the student"},
#         "question": {"type": "string", "description": "The question being asked"},
#         "question type" : {"type": "string",
#                            "enum": ["student grades", "student specific", "interest specific"],
#                            "description": "The type of question being asked"},
#         "interest" : {"type": "string", "description": "The interest of the student"},
#     },
#     "required": ["question", "question type"]
# }





# def get_tagging_chain(question)-> str:
#     global schema
#     chain = create_tagging_chain(schema, llm)
#     first_answer = chain.run(question)
#     print("first answer:")
#     print(first_answer)
#     return first_answer
#
#
# def get_grading_agent():
#
#     tools = [
#         Tool(
#             name="question type",
#             func=get_tagging_chain,
#             description="Useful when you need to understand the type of the input."
#         ),
#         StructuredTool(
#             name="Grade",
#             func=generate_answer,
#             description="Useful when you need to answer questions about students, grades, interests, etc from the context of canvas discussion posts. If the question is student specific, student name is required.",
#             args_schema=ToolArgsSchema
#         )
#     ]
#     # agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
#
#     agent = FakeAgent(output_parser=CustomOutputParser())
#     # prompt = CustomPromptTemplate(template=agent_prompt, tools=tools, input_variables=["input", "intermediate_steps"])
#     # output_parser = CustomOutputParser()
#     # tool_names = [tool.name for tool in tools]
#     # llm_chain = LLMChain(llm=llm, prompt=prompt)
#     # agent = LLMSingleActionAgent(
#     #     llm_chain=llm_chain,
#     #     output_parser=output_parser,
#     #     stop=["\nObservation:"],
#     #     allowed_tools=tool_names,
#     # )
#     agent_executor = AgentExecutor.from_agent_and_tools(
#         agent=agent, tools=tools, verbose=True
#     )
#
#     # return initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
#     return agent_executor
#
#
#
# def grade_answer(question) -> str:
#     global chat_history, vectorstore_index
#     agent = get_grading_agent()
#     return agent.run(question)