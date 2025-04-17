from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from KGAgents import KnowledgeGraphAgent
import os

# Set up Neo4j instance
n4j = Neo4jGraph(
    url="neo4j+s://your_URL",
    username="yourUsername",
    password="your_Password",
)
# Set up model
llama = ModelFactory.create(model_platform=ModelPlatformType.GROQ,
                            model_type=ModelType.GROQ_LLAMA_3_3_70B,
                            api_key=os.environ.get("GROQ_API_KEY"),
                            model_config_dict={"temperature": 0.4})

# Set instance
uio = UnstructuredIO()
kg_agent = KnowledgeGraphAgent(model=llama)

# Read content from input file
with open('sample_input.txt', 'r') as file:
    text_example = file.read()

# Create an element from given text
element_example = uio.create_element_from_text(text=text_example,
                                               element_id="0")

# Let Knowledge Graph Agent extract node and relationship information
ans_element = kg_agent.run(element_example, parse_graph_elements=False)
print(ans_element)
graph_elements = kg_agent.run(element_example, parse_graph_elements=True)
print(graph_elements)
# Add the element to neo4j database
n4j.add_graph_elements(graph_elements=[graph_elements])
