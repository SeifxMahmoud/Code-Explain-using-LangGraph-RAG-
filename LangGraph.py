#%%
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import Optional, List, Literal, Dict
import gradio as gr
import json
import faiss
import numpy as np
import traceback
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import ast
# %%
# State Definition
class AgentState(BaseModel):
    user_input: str
    intent: Optional[Literal['generate','explain']] = None
    retrieved_examples: Optional[List[Dict]] = None
    formatted_prompt: Optional[str] = None
    llm_output: Optional[str] = None
    final_response: Optional[str] = None
    

# %%
# RAG SetUP
base_folder = os.path.dirname(os.path.abspath(__file__))
data_path = 'C:/Users/sief x/Desktop/Study Sessions/RAG/human-eval/data/HumanEval.jsonl/human-eval-v2-20210705.jsonl'
with open(data_path, 'r', encoding = 'utf-8') as f:
    data = [json.loads(line) for line in f]    
# %%
prompts = [item['prompt'] for item in data]
task_ids = [item['task_id'] for item in data]
# %%
prompts
# %%
model_embed = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_embed.encode(prompts, show_progress_bar = True).astype('float32')
# %%
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# %%
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
model_llm = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-350M-mono')

# %%
# Helpers

def retrieve_similar_examples(query:str, k: int = 2) -> List[Dict]:
    '''
    This is the retrieval function for RAG where it takes the query and searches
    for most similar responses and returns the prompt+code.
    '''
    query_vec = model_embed.encode([query]).astype('float32')
    D,I = index.search(query_vec, k) # Because it uses L2Distance we pass a kn.
    return [
        {
            'task_id': task_ids[idx],
            'prompt': prompts[idx],
            'canonical_solution': data[idx]['canonical_solution']
            
        }
        for idx in I[0] 
    ]

# %%
def generation_prompt(retrieved_context: List[Dict], user_prompt: str) -> str:
    '''
    This is few-shot prompting to provide the LLM for the needed context 
    of the input and output examples before generating any new code.
    '''
    parts = ['### Examples: \n']
    for context in retrieved_context:
        parts.append(f"# Task: {context['prompt'].strip()}")
        parts.append(f"# Solution:\n{context['canonical_solution'].strip()}\n")
    parts.append('### New Task:')
    parts.append(f'# Task: {user_prompt.strip()}')
    parts.append('# Solution:')
    parts.append('def')
    return '\n'.join(parts)

# %%
def truncate_at_delimiter(text, delimiter = '# Task:'):
    '''
    This will split the output of the first task if the model decided to continue generating additional tasks. 
    '''
    return text.split(delimiter)[0].strip()


# %%
def generate_code(prompt_text: str,max_tokens = 128) -> str:
    '''
    takes the tokenized prompt from the user, then generates the new code. token -> text.
    '''
    inputs = tokenizer(prompt_text, return_tensors = 'pt')
    input_length = inputs['input_ids'].shape[1]
    outputs = model_llm.generate(
        **inputs,
        max_new_tokens = max_tokens,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.95,
        pad_token_id = tokenizer.eos_token_id or 50256 # Required by CodeGen to avoid errors related to padding. 
    )
    generated_tokens = outputs[0][input_length:] # this will slice off the prompt tokens, keeping only code generation.
    return truncate_at_delimiter(tokenizer.decode(generated_tokens, skip_special_tokens = True)) # Removes special tokens like 'pad' or 'eos'

# %%
# LangGraph Nodes
def start_node(state: AgentState) -> AgentState:
    '''
    This represents our starting node.
    '''
    return state
# %%
'''
This step is for intent classification using semantic similarity;
Defined the intent examples as it was before defined inside the function below;
this was problematic as the examples were embedded upon function call.
So we defined it outside and we only embed the user_input upon function call. 
'''
intent_examples = {
    'explain':[
        'please explain this code.',
        'tell me how this works.',
        'I want to understand this code.',
        'Can you break down the logic'
        ],
    'generate':[
        'Generate Python code for this.',
        'Write a function that does this',
        'Create a code that solves this problem',
        'Build a program for me',
        'Solve this using code'
            
        ]
        
    }
Intent_embeddings ={intent: model_embed.encode(examples, convert_to_tensor = True)
for intent, examples in intent_examples.items() }
   
def classify_intent(state: AgentState) -> AgentState:
    '''
    Instead of having an if-elif-else statement for checking the intent values;
    we now have somewhat of a classifier that checks by the score based on cosine simialrity between examples and user_input; more robust approach. 
    
    '''
    user_emb = model_embed.encode(state.user_input, convert_to_tensor = True)
    
    best_intent = None
    best_score = -1
    for intent, examples in Intent_embeddings.items():
        score = util.cos_sim(user_emb, examples).max().item()
        if score > best_score:
            best_score = score
            best_intent = intent
    if best_intent is None:
        raise ValueError('could Not Classify your Intent')
    state.intent = best_intent
    return state                   
# %%
def route(state:AgentState) -> str:
    '''
    takes in a state and returns the intent which is: explain, generate. 
    '''
    return state.intent


# %%

def explain_code(state: AgentState) -> AgentState:
    '''
    It is a function that takes the user's code, parses it using python's ast, trys then to give explanation for the user.
    '''
    user_text_lower = state.user_input.lower()
    trigger_keywords = ["explain", "solve the logic", "details about", "analyze"]

    # Default to entire input
    code_part = state.user_input

    # Extract only what's after the trigger keyword
    for keyword in trigger_keywords:
        if keyword in user_text_lower:
            idx = user_text_lower.find(keyword) + len(keyword)
            code_part = state.user_input[idx:]
            break
    code_part = code_part.strip()

    # Extra cleaning: remove prefixes like "this code:", "the code is", etc.
    prefixes_to_remove = ["this code:", "the code is", "here is the code:"]
    for prefix in prefixes_to_remove:
        if code_part.lower().startswith(prefix):
            code_part = code_part[len(prefix):].strip()
            break

    # If still contains non-code leading text, try to find where actual code starts
    for starter in ["def ", "class ", "import ", "from "]:
        if starter in code_part:
            code_part = code_part[code_part.find(starter):]
            break

    try:
        tree = ast.parse(code_part)
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
            func = tree.body[0]
            name = func.name
            args = [a.arg for a in func.args.args]
            explanation = (
                f"This code defines a function `{name}` that takes "
                f"{len(args)} parameter{'s' if len(args) != 1 else ''}: "
                f"{', '.join(f'`{a}`' for a in args)}, and "
            )
            if len(func.body) == 1 and isinstance(func.body[0], ast.Return):
                ret = func.body[0].value
                if isinstance(ret, ast.BinOp) and isinstance(ret.op, ast.Add):
                    explanation += "returns their sum using the `+` operator."
                else:
                    explanation += "returns the result of an expression."
            else:
                explanation += "contains multiple statements."
        else:
            explanation = (
                "This code is not a single function definition, "
                "so I'll describe it generally:\n\n"
                f"{code_part}"
            )
    except Exception:
        explanation = (
            "Could not parse the code, so here’s a general guess:\n\n"
            f"{code_part}\n\n"
            "☝️ You can improve the result by adding more context."
        )

    state.final_response = explanation
    return state

# %%
def generate_code_node(state: AgentState)-> AgentState:
    '''
    A function that takes the user input, prompts the LLM (codeGen) to generate new code.
    '''
    try: 
        retrieved = retrieve_similar_examples(state.user_input, k = 2)
        state.retrieved_examples = retrieved
        formatted = generation_prompt(retrieved, state.user_input)
        state.formatted_prompt = formatted
        llm_result = generate_code(formatted)
        state.llm_output = llm_result
        state.final_response = 'def' + llm_result
    except:
        state.final_response = "⚠️ Generation failed:\n" + str(e)
    return state
# %%
def end_node(state: AgentState) -> AgentState:
    '''
    simple end node
    '''
    return state
# %%
# ====== LANGGRAPH STRUCTURE ======

graph = StateGraph(AgentState)

graph.add_node("start", start_node)
graph.add_node("classify", classify_intent)
graph.add_node("generate", generate_code_node)
graph.add_node("explain", explain_code)
graph.add_node("end", end_node)

graph.set_entry_point("start")
graph.add_edge("start", "classify")
graph.add_conditional_edges("classify", route, {
    "generate": "generate",
    "explain": "explain"
})
graph.add_edge("generate", "end")
graph.add_edge("explain", "end")
graph.set_finish_point("end")

app = graph.compile()


# ====== GRADIO UI ======

def gradio_interface(user_input: str) -> str:
    try:
        initial_state = AgentState(user_input=user_input)
        final_state = app.invoke(initial_state)
        return final_state["final_response"]
    except Exception as e:
        return "❌ Error:\n" + traceback.format_exc()

gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=4, placeholder="Enter 'Generate a function...' or 'Explain this code: ...'"),
    outputs="text",
    title="🧠 LangGraph Code Assistant",
    description="Supports code generation and explanation using retrieval-augmented reasoning."
).launch()

# %%
def test_generate_add_function():
    input_text = "Generate a function that adds two numbers"
    output = gradio_interface(input_text)
    print("Input:", input_text)
    print("Output:", output)
    assert "def" in output.lower()
    assert "+" in output or "add" in output.lower()
    print("✅ Passed: test_generate_add_function")

def test_generate_sorting_function():
    input_text = "Generate a function that sorts a list"
    output = gradio_interface(input_text)
    print("Input:", input_text)
    print("Output:", output)
    assert "def" in output.lower()
    assert "sort" in output.lower()
    print("✅ Passed: test_generate_sorting_function")

def test_explain_loop_code():
    input_text = "Explain this code: for i in range(5): print(i)"
    output = gradio_interface(input_text)
    print("Input:", input_text)
    print("Output:", output)
    assert any(keyword in output.lower() for keyword in ["loop", "iterates", "range", "prints"])
    print("✅ Passed: test_explain_loop_code")


def test_invalid_input():
    input_text = "What's the weather in Cairo?"
    try:
        output = gradio_interface(input_text)
    except Exception as e:
        print("✅ Properly raised error:", str(e))
        return
    print("Output:", output)
    assert "❌" in output or "could not classify" in output.lower()
    print("✅ Passed: test_invalid_input")

# %% Visualization for LangGraph
app.get_graph().print_ascii()
# %%
test_generate_add_function()
test_generate_sorting_function()
test_explain_loop_code()
test_invalid_input()

# %%
