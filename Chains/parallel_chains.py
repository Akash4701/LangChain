from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel


load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0
)

prompt1=PromptTemplate(
    template="You are a helpful assistant.Genrate best notes {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(template="Genrate five question and answer quiz on the :{topic}",
    input_variables=["topic"])
prompt3=PromptTemplate(template="Genrate five question and answer quiz on the :{topic}",
    input_variables=["topic"])

prompt3 =PromptTemplate(template="Merge the quiz and the notes on the single document:{quiz} and {notes}",
                        input_variables=['quiz','notes'])

parsers=StrOutputParser()
parallel_chain=RunnableParallel({
    "notes": prompt1 | model | parsers,
    "quiz": prompt2 | model | parsers
})

merge_chain= prompt3 | model | parsers

chain=parallel_chain|merge_chain


text="""Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that best separates data points of different classes in a high-dimensional space. SVM aims to maximize the margin between the data points and the hyperplane, which helps improve generalization and reduces the risk of overfitting. For cases where data is not linearly separable, SVM uses kernel functions (like polynomial or radial basis function) to transform the input space into a higher dimension where separation is possible. Due to its effectiveness in high-dimensional spaces and robustness against overfitting, SVM is widely used in applications like image recognition, text classification, and bioinformatics.
"""

result=chain.invoke({"topic":text})

print(result)

chain.get_graph().print_ascii()