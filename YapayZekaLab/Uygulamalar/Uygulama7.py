from random import choice
from experta import*

class Light(Fact):
    """   """
    pass

class CrossOver(KnowledgeEngine):
    @Rule(Light(renk="green"))
    def greenLight(self):
        print("you can walk")

    @Rule(Light(renk="red"))
    def redLight(self):
        print("wait")

    @Rule(Light(renk="yellow"))
    def yellowLight(self):
        print("get ready")

#%%
uzman = CrossOver()
uzman.reset()
uzman.declare(Light(renk=choice(["green", "yellow", "red"])))
uzman.run()
#%%
engine = CrossOver()
engine.reset()
engine.declare(Light(color=choice(['green', 'yellow', 'red'])))
engine.run()
