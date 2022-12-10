import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Girdilerin ve ciktilarin tanimlanmasi
kalite = ctrl.Antecedent(np.arange(0,11,1), "kalite") # np.arange(included, not inclueded, space)
servis = ctrl.Antecedent(np.arange(0,11,1), "servis")
bahsis = ctrl.Consequent(np.arange(0,26,1), "bahsis")
# Kalite ve servis girdileri icin uyelik fonksiyonlarinin belirlenmesi
kalite.automf(3)
servis.automf(3)
# Cikti degeri icin uyelik fonksiyonun belirlenmesi
bahsis["dusuk"] = fuzz.trimf(bahsis.universe, [0,0,13]) # [begin, highest, end] triangular graph
bahsis["orta"] = fuzz.trimf(bahsis.universe, [0,13,25])
bahsis["yuksek"] = fuzz.trimf(bahsis.universe, [13,25,25])
# kalite girdisine ait uyelik fonksiyonlarinin gorsel olarak incelenmesi
kalite.view()
# servis girdisine ait uyelik fonksiyonlarinin gorsel olarak incelenmesi
servis.view()
# bahsis ciktisina ait uyelik fonksiyonlarinin gorsel olarak incelenmesi
bahsis.view()
# Bulanik kurallarinin belirlenmesi
kural1 = ctrl.Rule(kalite["good"] | servis["good"], bahsis["yuksek"])
kural2 = ctrl.Rule(servis["average"], bahsis["orta"])
kural3 = ctrl.Rule(kalite["poor"] | servis["poor"], bahsis["dusuk"])
# Bahsisin belirlenmesi
bahsisKontrol = ctrl.ControlSystem([kural1, kural2, kural3])
bahsisBelirleme = ctrl.ControlSystemSimulation(bahsisKontrol)
# Bahsisin hesaplanmasi
bahsisBelirleme.input["kalite"] = 3.2
bahsisBelirleme.input["servis"] = 2.4
bahsisBelirleme.compute()
print(bahsisBelirleme.output["bahsis"])
# Bahsisin gorsel olarak gosterilmesi
bahsis.view(sim=bahsisBelirleme)

#%%
