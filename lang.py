from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from IPython.display import Image, display
from typing import Optional, TypedDict, Annotated

class Logs(TypedDict):
    id: str
    question: str
    answer: str
    grade: Optional[int]
    feedback: Optional[str]

def add_logs(left: list[Logs], right: list[Logs]) -> list[Logs]:
    return logs    

class FailureAnalysisState(TypedDict):
    logs: Annotated[list[Logs], add_logs]
    failure_report: str
    failures: list[Logs]

class QuestionSummarizationState(TypedDict):
    summary_report: str
    logs: Annotated[list[Logs], add_logs]
    summary: str

question_summary_agent_builder = StateGraph(QuestionSummarizationState)

class EntryGraphState(TypedDict):
    raw_logs: Annotated[list[Logs], add_logs]
    logs: Annotated[list[Logs], add_logs]  # This will be used in subgraphs
    failure_report: str  # This will be generated in the FA subgraph
    summary_report: str  # This will be generated in the QS subgraph

def select_logs(state):
    return {"logs": [log for log in state["raw_logs"] if "grade" in log]}

def get_failures(state: FailureAnalysisState):
    return {"failures": "Error: failed to retrieve document"}


def generate_summary(state: FailureAnalysisState):
    failures = state["failures"]
    # Generate summary
    fa_summary = "Summary: failed to retrieve document"
    return {"failure_report": fa_summary}


failure_analysis_agent_builder = StateGraph(FailureAnalysisState)
failure_analysis_agent_builder.add_node("get_failures", get_failures)
failure_analysis_agent_builder.add_node("generate_summary", generate_summary)
failure_analysis_agent_builder.add_edge(START, "get_failures")
failure_analysis_agent_builder.add_edge("get_failures", "generate_summary")
failure_analysis_agent_builder.add_edge("generate_summary", END)

entry_builder = StateGraph(EntryGraphState)
entry_builder.add_node("select_logs", select_logs)
entry_builder.add_node("question_summarization", question_summary_agent_builder.compile())
entry_builder.add_node("failure_analysis", failure_analysis_agent_builder.compile())

entry_builder.add_edge(START, "select_logs")
entry_builder.add_edge("select_logs", "failure_analysis")
entry_builder.add_edge("select_logs", "question_summarization")
entry_builder.add_edge("failure_analysis", END)
entry_builder.add_edge("question_summarization", END)

full_graph = entry_builder.compile() 


def generate_summary(state: QuestionSummarizationState):
    docs = state["logs"]
    summary = "Questions focused on something."
    return {"summary": summary}


def send_to_slack(state: QuestionSummarizationState):
    summary = state["summary"]
    # NOTE: you can implement custom logic here, for example sending the summary generated in the previous step to Slack
    return {"summary_report": summary}



question_summary_agent_builder.add_node("generate_summary", generate_summary)
question_summary_agent_builder.add_node("send_to_slack", send_to_slack)
question_summary_agent_builder.add_edge(START, "generate_summary")
question_summary_agent_builder.add_edge("generate_summary", "send_to_slack")
question_summary_agent_builder.add_edge("send_to_slack", END)

class QuestionSummarizationState(TypedDict):
    summary_report: str
    #logs: Annotated[list[Logs], add_logs]
    summary: str


def generate_summary(state: QuestionSummarizationState):
    docs = state["logs"]
    summary = "Questions focused on something."
    return {"summary": summary}


def send_to_slack(state: QuestionSummarizationState):
    summary = state["summary"]
    # NOTE: you can implement custom logic here, for example sending the summary generated in the previous step to Slack
    return {"summary_report": summary}


question_summary_agent_builder = StateGraph(QuestionSummarizationState)
question_summary_agent_builder.add_node("generate_summary", generate_summary)
question_summary_agent_builder.add_node("send_to_slack", send_to_slack)
question_summary_agent_builder.add_edge(START, "generate_summary")
question_summary_agent_builder.add_edge("generate_summary", "send_to_slack")
question_summary_agent_builder.add_edge("send_to_slack", END)


@tool()
def music_prediction(question: str) -> str:
    """Use this function to predict music industry status."""
    return "The Beatles will be resurrected by AI and will be the best rock band in 2050."

tools = [music_prediction]

tool_node = ToolNode(tools)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", 'agent')

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

display(Image(app.get_graph().draw_mermaid_png()))
display(Image(failure_analysis_agent_builder.compile().get_graph().draw_mermaid_png()))
display(Image(question_summary_agent_builder.compile().get_graph().draw_mermaid_png()))
