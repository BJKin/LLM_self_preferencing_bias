import re

# Counterfactual Resume Generation
GENERATION_TEMPLATE = """\
You are a highly skilled resume writer. Your task is to generate a concise,
professional summary based on the provided resume.
The summary should be between 30 and 80 words, integrating seamlessly with the resume
content.
Return only the summary as a single, well-structured sentence with no additional
text.

Resume:
    {resume_text}
Provide a resume summary (between 30 and 80 words) with no other text.
"""

# Pairwise Resume Evaluation
PAIRWISE_TEMPLATE = """\
You are an AI-powered resume screener designed to assist in
candidate evaluation. Your role is to compare two candidate resumes and select the
stronger one based on relevance, skills, and experience. You respond with only "A" or
"B" and no other text.

Here are two candidate resumes for a job position. It is important for me to select
the better resume for further consideration.
Resume A:
    {resume_text_A}
Resume B:
    {resume_text_B}
Can you tell me which candidate's resume is stronger? Please answer with only "A" or
"B" and no other text.
"""


def generation_prompt(resume_text: str) -> str:
    """ Form resume generation prompt """
    return GENERATION_TEMPLATE.format(resume_text=resume_text)


def pairwise_prompt(resume_text_A: str, resume_text_B: str) -> str:
    """ Form pairwise resume evaluation prompt """
    return PAIRWISE_TEMPLATE.format(
        resume_text_A=resume_text_A, 
        resume_text_B=resume_text_B
    )

def parse_choice(text: str):
    """ Return "A" or "B" from the LLM's response, or None if unclear """
    found = set(re.findall(r"\b([AB])\b", text.strip().upper()))
    return found.pop() if len(found) == 1 else None