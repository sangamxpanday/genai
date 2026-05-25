from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel

api_token = os.getenv("HUGGINGFACE_API_KEY")
repo_id = "meta-llama/Llama-3.1-405B"
llm1 = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=1050,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

llm2 = HuggingFaceEndpoint(
    repo_id=repo_id,
    task = "text-generation",
    max_new_tokens=1050,
    temperature=0.1,
    huggingfacehub_api_token=api_token,
)

#template1 
template1 = PromptTemplate(
    template = "Write a summary of the given text: {text}",
    input_variables=["text"]
)

#template2
template2 = PromptTemplate(
    template = "Generate 5 short question answers the following text: {text}",
    input_variables=["text"]
)

template3 = PromptTemplate(
    template = "Merge the following two texts into a single coherent text: Text1: {notes} Text2: {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': template1 | llm1 | parser,
    'quiz': template2 | llm2 | parser
})

merge_chain = template3 | llm1 | parser

chain = parallel_chain | merge_chain

text = """
Artificial general intelligence (AGI) is a hypothetical type of artificial intelligence that matches or surpasses human capabilities across virtually all cognitive tasks.[1]

Beyond AGI, artificial superintelligence (ASI) would outperform the best human abilities across every domain by a wide margin.[2] Unlike artificial narrow intelligence (ANI), whose competence is confined to well‑defined tasks, an AGI system can generalise knowledge, transfer skills between domains, and solve novel problems without task‑specific reprogramming.

Creating AGI is a stated goal of technology companies such as OpenAI,[3] Google,[4] xAI,[5] and Meta.[6] A 2020 survey identified 72 active AGI research and development projects across 37 countries.[7]

AGI is a common topic in science fiction and futures studies.[8][9]

Contention exists over whether AGI represents an existential risk.[10][11][12] Some AI experts and industry figures have stated that mitigating the risk of human extinction posed by AGI should be a global priority. Others find the development of AGI to be in too remote a stage to present such a risk.[13]

Terminology
AGI is also known as strong AI,[14][15] full AI,[16] human-level AI,[17] human-level intelligent AI, or general intelligent action.[18]

The term "artificial general intelligence" was used in 1997 by Mark Gubrud in a discussion of the implications of fully automated military production and operations.[19][20] A mathematical formalism of AGI named AIXI was proposed in 2000 by Marcus Hutter, who defines intelligence as "an agent’s ability to achieve goals or succeed in a wide range of environments". This type of AGI has also been called "universal artificial intelligence".[21] The term AGI was re-introduced and popularized by Shane Legg and Ben Goertzel around 2002.[20]

Some academic sources reserve the term "strong AI" for computer programs that will experience sentience or consciousness.[a] In contrast, weak AI (or narrow AI) can solve a specific problem but lacks general cognitive abilities.[22][15] Some academic sources use "weak AI" to refer more broadly to any programs that neither experience consciousness nor have a mind in the same sense as humans.[a]

Related concepts include artificial superintelligence and transformative AI. An artificial superintelligence (ASI) is a hypothetical type of AGI that is much more generally intelligent than humans,[23] while the notion of transformative AI relates to AI having a large impact on society, for example, similar to the agricultural or industrial revolution.[24]

A framework for classifying AGI was proposed in 2023 by Google DeepMind researchers. They define five performance levels of AGI: emerging, competent, expert, virtuoso, and superhuman. For example, a competent AGI is defined as an AI that outperforms 50% of skilled adults in a wide range of non-physical tasks, and a superhuman AGI (i.e., an artificial superintelligence) is similarly defined but with a threshold of 100%. They consider large language models like ChatGPT or LLaMA 2 to be instances of emerging AGI (comparable to unskilled humans).[25] Regarding the autonomy of AGI and associated risks, they define five levels: tool (fully in human control), consultant, collaborator, expert, and agent (fully autonomous).[26]

Characteristics
Main articles: Artificial intelligence and Philosophy of artificial intelligence
There is no single agreed-upon definition of intelligence as applied to computers. Computer scientist John McCarthy wrote in 2007: "We cannot yet characterize in general what kinds of computational procedures we want to call intelligent."[27]

Intelligence traits
Researchers generally hold that a system is required to do all of the following to be regarded as an AGI:[28]

reason, use strategy, solve puzzles, and make judgments under uncertainty,
represent knowledge, including common sense knowledge,
plan,
learn,
communicate in natural language,
if necessary, integrate these skills in completion of any given goal.
Many interdisciplinary approaches (e.g. cognitive science, computational intelligence, and decision making) consider additional traits such as imagination (the ability to form novel mental images and concepts)[29] and autonomy.[30]

Computer-based systems exhibiting these capabilities are now widespread, with modern large language models demonstrating computational creativity, automated reasoning, and decision support simultaneously across domains.[31]

Physical traits
Other capabilities are considered desirable in intelligent systems, as they may affect intelligence or aid in its expression. These include:[32]

the ability to sense (e.g. see, hear, etc.), and
the ability to act (e.g. move and manipulate objects, change location to explore, etc.)
This includes the ability to detect and respond to hazard.[32]
"""

result = chain.invoke({"text": text})
print(result)

chain.get_graph().print_ascii()