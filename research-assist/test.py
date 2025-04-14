from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# output format
class Review(BaseModel):
    summary: str = Field(description="Summarise the given text in brief, two to three lines")
    sentiment: Literal["pos", "neg"] = Field(description="This field should contain the sentiment of the given text," \
    "\"pos\" for positive sentiment and \"neg\" for negative sentiment")

# parser
parser = JsonOutputParser(pydantic_object=Review)

# template and prompt
template = PromptTemplate(
    template="Generate summary and sentiment of the given text \{text} \n{format}",
    input_variables=["text"],
    partial_variables={"format": parser.get_format_instructions()}
)
prompt = template.invoke({"text":"I was genuinely disappointed with the latest iPhone. Despite the hype, the battery drains way too fast,"
"and the device overheats with minimal usage. The camera quality, which was supposed to be a major upgrade, shows little to no"
"improvement over previous models. To make matters worse, the phone is ridiculously overpriced for the features it offers,"
"and the lack of a charger in the box just adds to the frustration. Overall, it's a letdown for such a premium product."})

# model
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)
response = model.invoke(prompt)
print(f"{parser.parse(response.content)}")
