# langsmith_instrument.py
import os
from langsmith import LangSmithClient  # pseudocode; real import may differ
client = LangSmithClient(api_key=os.environ["LANGSMITH_API_KEY"])

def trace_example():
    trace = client.start_trace(name="local-debug-trace")  # pseudocode
    span = trace.start_span("do_work")
    # call your function / web request etc.
    do_your_work()
    span.end()
    trace.finish()