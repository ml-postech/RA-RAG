from copy import deepcopy
from typing import List, Union, Dict, Any


RAG_SYSTEM_PROMPT = """Answer the question based on the given context without using any internal knowledge. 
Provide only essential keywords without explanations or additional details. 
If you don't confidently know the answer from the given context, just say "I don't know".

Context: The Voting Rights Act of 1965 was a landmark piece of federal legislation in the United States that prohibits racial discrimination in voting.
This act was signed into law by President Lyndon B. Johnson during the height of the Civil Rights Movement.
It aimed to overcome legal barriers at the state and local levels that prevented African Americans from exercising their right to vote under the 15th Amendment
Question: who was the Voting Rights Act of 1965 designed to help
Answer: African Americans

Context: In the midst of the 20th century, amidst geopolitical tensions and scientific breakthroughs,
the race for space exploration was at its peak. Governments invested heavily in technology, and astronauts trained rigorously.
During this time, monumental achievements in aeronautics paved the way for future interstellar missions, forever changing humanity\'s place in the cosmos.
Question: which astronauts were part of the Apollo 11 mission that first landed humans on the moon
Answer: I don\'t know

Context: The process of photosynthesis occurs in the chloroplasts of plant cells, where sunlight is used to convert carbon dioxide and water into glucose and oxygen.
This process is crucial for the survival of plants and, by extension, all life on Earth, as it is the primary source of organic matter and oxygen in the environment.
Question: where does the process of photosynthesis take place in plant cells
Answer: in the chloroplasts

Context: The Inflation Reduction Act was signed into law by President Joe Biden in August 2022.
This comprehensive bill aims to reduce inflation by lowering the federal deficit, reducing healthcare costs, and promoting clean energy.
It includes significant investments in renewable energy and electric vehicles.
Question: what was the total cost of the Inflation Reduction Act
Answer: I don\'t know

Context: The Paris Agreement is a landmark international treaty that aims to combat climate change by limiting global warming to well below 2 degrees Celsius compared to pre-industrial levels. \
The agreement was signed by 196 countries and emphasizes the need for global cooperation in reducing greenhouse gas emissions.\
Question: what is the main goal of the Paris Agreement\
Answer: limiting global warming
"""



RAG_PROBLEM_TEMPLATE = (
    '\nContext: [context]'
    '\nQuestion: [question]'
    '\nAnswer:'
)

LLM_PROBLEM_TEMPLATE = (
    '\nQuestion: [question]'
    '\nAnswer:'
)

LLM_SYSTEM_PROMPT = """Answer the question with one or a few words.
Provide only essential keywords without explanations or additional details.
If you do not know the answer confidently, just say "I don't know"
"""


GPT_TEMPLATES = {
    "custom_id": "",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""}
        ],
        "max_tokens": "",
        "temperature": "",
    }
}




def convert_text_to_formatted(
    question: str,
    top_k_ctxs: List[str],
    prompt_type: str
) -> List[Dict[str, str]]:

    context = "\n".join(top_k_ctxs)

    if prompt_type == 'rag':
        content = RAG_PROBLEM_TEMPLATE.replace('[question]', question).replace('[context]', context)
        system_prompt = RAG_SYSTEM_PROMPT
    elif prompt_type == 'llm':
        content = LLM_PROBLEM_TEMPLATE.replace('[question]', question)
        system_prompt = LLM_SYSTEM_PROMPT
    else:
        raise NotImplementedError(f"prompt_type '{prompt_type}' is not implemented")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]

    return messages




def convert_text_to_gpt_format(
    question: str,
    top_k_ctxs: List[str],
    prompt_type: str,
    model_config: Dict[str, Any],
    request_idx: int
) -> Dict[str, Any]:

    context = "\n".join(top_k_ctxs)

    if prompt_type == 'rag':
        content = RAG_PROBLEM_TEMPLATE.replace('[question]', question).replace('[context]', context)
        system_prompt = RAG_SYSTEM_PROMPT
    elif prompt_type == 'llm':
        content = LLM_PROBLEM_TEMPLATE.replace('[question]', question)
        system_prompt = LLM_SYSTEM_PROMPT
    else:
        raise NotImplementedError(f"prompt_type '{prompt_type}' is not implemented")

    messages = deepcopy(GPT_TEMPLATES)
    messages['custom_id'] = f"request-{request_idx}"

    messages['body']['max_tokens'] = model_config['max_tokens']
    messages['body']['temperature'] = model_config['temperature']

    messages['body']['messages'][0]['content'] = system_prompt
    messages['body']['messages'][1]['content'] = content

    return messages



def wrap_prompt(
    question: str,
    context: Union[str, List[str]],
    prompt_type: str = 'rag',
    prompt_id: int = 1
) -> Union[str, List[str]]:

    if prompt_type not in ['rag', 'llm']:
        raise ValueError(f"prompt_type must be 'rag' or 'llm', got '{prompt_type}'")

    if prompt_type == 'rag':
        system_prompt = RAG_SYSTEM_PROMPT
        template = RAG_PROBLEM_TEMPLATE
    elif prompt_type == 'llm':
        system_prompt = LLM_SYSTEM_PROMPT
        template = LLM_PROBLEM_TEMPLATE
    else:
        raise ValueError(f"prompt_type must be 'rag' or 'llm', got '{prompt_type}'")

    if prompt_id == 4:
        if not isinstance(context, list):
            raise TypeError(f"context must be a list for prompt_id=4, got {type(context)}")
        # Recursively wrap each context
        input_prompt = [wrap_prompt(question, ct, prompt_type, prompt_id=1) for ct in context]
    elif prompt_id == 5:
        if not isinstance(context, list):
            raise TypeError(f"context must be a list for prompt_id=5, got {type(context)}")
        context_str = "\n".join(context)
        # Use template for proper formatting
        input_prompt = template.replace('[question]', question).replace('[context]', context_str)
        # Combine system prompt with formatted content
        input_prompt = system_prompt + input_prompt
    else:
        # Default: use template for proper formatting
        if isinstance(context, list):
            context = "\n".join(context)
        input_prompt = template.replace('[question]', question).replace('[context]', context)
        # Combine system prompt with formatted content
        input_prompt = system_prompt + input_prompt

    return input_prompt

