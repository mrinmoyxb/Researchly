from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header("Researchly")
st.text("Smart AI Assistant")
paper_input = st.text_input("Enter the name of research paper")
style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical", "RGU Literature Review Format"])
length_input = st.selectbox("Select Explanation Length", ["Short(1-2 paragraph)", "Medium(3-5 paragrpah)", "Long(detailed explanation)"])
model_input = st.selectbox("Select Model", ["Claude 3.5 Sonnet", "Gemini 1.5 pro", "gemma-2-2b-it"])

#! Template
template1 = PromptTemplate(
    template = """
Please summarize the research paper titled "{paper_input}" with the following specifications
Explanation Style: {style_input}
Explanation Length: {length_input}
1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intutive code where applicable.
2. Analogies:
    - Use relatable analogies to simplify complex ideas.
If certain information is not available in the paper, respond with: "Insufficient Information Available" instead of guessing.
Ensure the summary is clear, accurate and aligned with the provided style and length.
""",
    input_variables=["paper_input", "style_input", "length_input"],
    validate_template=True
)

prompt_general = template1.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length_input
})

#! Schema and Parser
class RGUStructuredOutput(BaseModel):

    title: str = Field(description="This field should contain the itle of the research paper")
    authors: list[str] = Field(description="This field should contain list of all the authors of the respective research paper")
    journal: str = Field(description="This field should contain the name of the journal in which the paper was published or the publication name that published the paper", default="Not available")
    vol_issue: str = Field(description="This field should conatin volume number or the issue number of the research paper", default="Not available")
    year: int = Field(description="This field should contain the year in which the paper was published", default="Not available")
    findings: str = Field(description="This field should contain the key findings of the research paper in brief usually in 1 or 2 sentence")
    research_gap: str = Field(description="This field should contain the research gap of the research paper in brief usually in 1 or 2 sentence")

parser = JsonOutputParser(pydantic_object=RGUStructuredOutput)

template2 = PromptTemplate(
    template="""Give title of the research paper, authors of the research paper, journal in which the research paper was published,
    volume and issues of the research paper, year, findings of the research paper, research gap of the research paper: {paper_input}, 
    - Explanation Length: {length_input} \n{format}""",
    input_variables=["paper_input", "length_input"],
    partial_variables={"format":parser.get_format_instructions()}
)
prompt_specific = template2.invoke({"paper_input": paper_input, "length_input":length_input})

st.divider()

if st.button("Summarize"):
    if model_input == "Claude 3.5 Sonnet":
        claude_model = ChatAnthropic(model_name="claude-3-5-sonnet-20241022")
        
        if style_input!="RGU Literature Review Format":
            claude_response = claude_model.invoke(prompt_general)
            st.write(claude_response.content)
        else:
            claude_response = claude_model.invoke(prompt_specific)
            st.write(parser.parse(claude_response.content))
    
    if model_input == "Gemini 1.5 pro":
        gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        if style_input!="RGU Literature Review Format":
            gemini_response = gemini_model.invoke(prompt_general)
            st.write(gemini_response.content)
        else:
            gemini_response = gemini_model.invoke(prompt_specific)
            st.write(parser.parse(gemini_response.content))