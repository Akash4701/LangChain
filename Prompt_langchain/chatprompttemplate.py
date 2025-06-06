from langchain_core.prompts import ChatPromptTemplate


chat_template=ChatPromptTemplate(
    [
        ('system','you are a {domain} expert'),
        ('user','you are a {role} genius')
    ])

result=chat_template.invoke({'domain':'Cricket','role':'umpire'})
print(result)