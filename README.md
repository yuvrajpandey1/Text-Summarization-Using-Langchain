
# Summarize Text

Suppose you have any type of file in any format and doesn't matter whether it is Structured or Unstructured and you want to summarize the content.

LLMs are a great tool for this given their proficiency in understanding and synthesizing text.In the context of RAG, summarizing text can help distill the information in a large number of retrieved documents to provide context for a LLM.

In this we'll go over how to summarize content from multiple documents using LLMs.


# Documentation 

[Summarization](https://python.langchain.com/v0.1/docs/use_cases/summarization/)

A central question for building a summarizer is how to pass your documents into the LLM's context window. Two common approaches for this are:

* StuffDocumentsChain: This chain takes a list of documents and first combines them into a single string. It does this by formatting each document into a string with the document_prompt and then joining them together with document_separator. It then adds that new string to the inputs with the variable name set by document_variable_name. Those inputs are then passed to the llm_chain.

* Map-Reduce: The ReduceDocumentsChain handles taking the document mapping results and reducing them into a single output. It wraps a generic CombineDocumentsChain (like StuffDocumentsChain) but adds the ability to collapse documents before passing it to the CombineDocumentsChain if their cumulative size exceeds token_max. In this example, we can actually re-use our chain for combining our docs to also collapse our docs.

So if the cumulative number of tokens in our mapped documents exceeds 4000 tokens, then we'll recursively pass in the documents in batches of < 4000 tokens to our StuffDocumentsChain to create batched summaries. And once those batched summaries are cumulatively less than 4000 tokens, we'll pass them all one last time to the StuffDocumentsChain to create the final summary.

* RefineDocumentsChain: This chain collapses documents by generating an initial answer based on the first document and then looping over the remaining documents to refine its answer. This operates sequentially, so it cannot be parallelized. It is useful in similar situations as MapReduceDocuments Chain, but for cases where you want to build up an answer by refining the previous answer (rather than parallelizing calls).







 








## Important Libraries Used

 - [LLMChain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html)
 - [Prompt Template](https://python.langchain.com/v0.1/docs/integrations/llms/deepinfra/#create-a-prompt-template)
- [ChatGroq](https://python.langchain.com/docs/integrations/chat/groq/)
 - [pypdf](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.PyPDFLoader.html)






## Plateform or Providers

 - [LangChain-ChatGroq](https://python.langchain.com/docs/integrations/chat/groq/)
 - [Streamlit](https://docs.streamlit.io/)

## Model

 - LLM - Gemma-7b-It


## Installation

Install below libraries

```bash
  pip install langchain
  pip install langchain_community
  pip install streamlit
  pip install langchain-groq
  pip install pypdf

```
    
## Tech Stack

**Client:** Python, LangChain PromptTemplate, ChatGroq

**Server:** Anaconda Navigator, Jupyter Notebook


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`GROQ_API_KEY`


## Examples
Load Documents
```javascript
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
```

## Insert them all into a prompt, and pass that prompt to an LLM

```javascript
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise summary of the following:\\n\\n{context}")]
)

# Instantiate chain
chain = create_stuff_documents_chain(llm, prompt)

# Invoke chain
result = chain.invoke({"context": docs})
print(result) 
```

