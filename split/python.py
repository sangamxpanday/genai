from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
    class Robot:
    # The __init__ function sets up the object's data (attributes)
    def __init__(self, name, task):
        self.name = name
        self.task = task

    # A custom function (method) that performs an action
    def introduce(self):
        return f"Hi, I am {self.name} and my job is {self.task}."

# Creating an 'instance' (object) of the Robot class
my_robot = Robot("Sparky", "cleaning")

# Calling the function inside the class
print(my_robot.introduce())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)