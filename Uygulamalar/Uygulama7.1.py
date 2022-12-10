from random import choice
from experta import *

############## 170541028 ABDULLAH YILDIZ

class Doctor(Fact):
    """   """
    pass


class goToDoctor(KnowledgeEngine):
    # dis fircalarken dis eti kanamasi olursa
    @Rule(Doctor(issue="short bleeding"))
    def tissueBleeding(self):
        print("dis hastaligi vardir ve dis hekimine basvur")

    # dis fircalarken uzun sureli dis eti kanamasi olursa
    @Rule(Doctor(issue="too much bleeding"))
    def tooMuchTissueBleeding(self):
        print("diseti cekilmesi vardir ve dis hekimine basvur")

    # eger dis eti cekilmesi var ve dis koku gorunuyorsa
    @Rule(Doctor(issue="teeth root visible"))
    def isTeethRootVisible(self):
        print("dolgu yaptir")

    # diste yiyecek ve iceceklerden olusan renk degisimi varsa
    @Rule(Doctor(issue="color change"))
    def cleanTeeth(self):
        print("disleri temizle")

    # yeni dis cikarirken morarma gorunuyorsa
    @Rule(Doctor(issue="it's purple"))
    def isItPurple(self):
        print("dis hekimine basvur")

    # diste agri yapmayan curuk varsa
    @Rule(Doctor(issue="rotten teeth"))
    def fillUp(self):
        print("dolgu yaptir")

    # disteki curuk ileri derecedeyse
    @Rule(Doctor(issue="severe rotten teeth"))
    def canalCure(self):
        print("kanal tedavisi ve dolgu yaptir")


uzman = goToDoctor()

# %%
uzman.reset()
uzman.declare(Doctor(issue=choice(['short bleeding', 'too much bleeding', 'teeth root visible',
                                   'color change', 'it\'s purple', 'rotten teeth', 'severe rotten teeth'])))
uzman.run()
