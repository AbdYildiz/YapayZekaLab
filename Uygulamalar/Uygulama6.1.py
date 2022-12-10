import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Girdilerin tanimlanmasi
dishes = ctrl.Antecedent(np.arange(0, 101, 1), "dishes")  # np.arange(included, not inclueded, space)
dirty = ctrl.Antecedent(np.arange(0, 101, 1), "dirty")
type = ctrl.Antecedent(np.arange(0, 101, 1), "type")
# Ciktilarin tanimlanmasi
time = ctrl.Consequent(np.arange(30, 161, 10), "Time")
detergentAmount = ctrl.Consequent(np.arange(0, 101, 1), "Amount")
temperature = ctrl.Consequent(np.arange(35, 76, 1), "Temperature")
upperPump = ctrl.Consequent(np.arange(2100, 3500, 100), "Upper Pump")
lowerPump = ctrl.Consequent(np.arange(2100, 3500, 100), "Lower Pump")

# dishes, dirty ve type girdileri icin uyelik fonksiyonlarinin belirlenmesi
dishes.automf(3)  # poor, average, good
dirty.automf(3)
type.automf(3)

# Time cikti degeri icin uyelik fonksiyonun belirlenmesi
time["lower"] = fuzz.trimf(time.universe, [30, 30, 60])  # [begin, highest, end] triangular graph
time["low"] = fuzz.trimf(time.universe, [40, 65, 90])
time["average"] = fuzz.trimf(time.universe, [70, 95, 120])
time["high"] = fuzz.trimf(time.universe, [100, 125, 150])
time["higher"] = fuzz.trimf(time.universe, [130, 160, 160])

# Temperature cikti degeri icin uyelik fonksiyonun belirlenmesi
temperature["poor"] = fuzz.trimf(temperature.universe, [35, 35, 50])  # [begin, highest, end] triangular graph
temperature["average"] = fuzz.trimf(temperature.universe, [37.5, 52.5, 67.5])
temperature["good"] = fuzz.trimf(temperature.universe, [55, 76, 76])
# dishes girdisine ait uyelik fonksiyonlarinin gorsel olarak incelenmesi
# dishes.view()
# dirty girdisine ait uyelik fonksiyonlarinin gorsel olarak incelenmesi
# time.view()
# bahsis ciktisina ait uyelik fonksiyonlarinin gorsel olarak incelenmesi
# temperature.view()
# Bulanik kurallarinin belirlenmesi
kural1 = ctrl.Rule(dishes["poor"] | dirty["poor"] | type["poor"], temperature["poor"])
kural2 = ctrl.Rule(dishes["poor"] | dirty["good"] | type["average"], temperature["good"])
kural3 = ctrl.Rule(dishes["average"] | dirty["average"] | type["good"], temperature["average"])
kural4 = ctrl.Rule(dishes["good"] | dirty["good"] | type["average"], temperature["good"])
# temperature belirlenmesi
temperatureControl = ctrl.ControlSystem([kural1, kural2, kural3, kural4])
temperatureDecide = ctrl.ControlSystemSimulation(temperatureControl)
# temperature hesaplanmasi
temperatureDecide.input["dishes"] = 62
temperatureDecide.input["dirty"] = 40
temperatureDecide.input["type"] = 89
temperatureDecide.compute()
print("Temperature", temperatureDecide.output["Temperature"])
# temperature gorsel olarak gosterilmesi
temperature.view(sim=temperatureDecide)

# time belirlenmesi
kural1 = ctrl.Rule(dishes["poor"] | dirty["poor"] | type["poor"], time["lower"])
kural2 = ctrl.Rule(dishes["poor"] | dirty["good"] | type["average"], time["average"])
kural3 = ctrl.Rule(dishes["average"] | dirty["average"] | type["good"], time["average"])
kural4 = ctrl.Rule(dishes["good"] | dirty["good"] | type["average"], time["higher"])
# time belirlenmesi
timeControl = ctrl.ControlSystem([kural1, kural2, kural3, kural4])
timeDecide = ctrl.ControlSystemSimulation(timeControl)
# time hesaplanmasi
timeDecide.input["dishes"] = 62
timeDecide.input["dirty"] = 40
timeDecide.input["type"] = 89
timeDecide.compute()
print("Time : ", timeDecide.output["Time"])
# temperature gorsel olarak gosterilmesi
time.view(sim=timeDecide)

######################################################################################
# Hocam kural tanimlarini yaparken
# ctrl.Rule(dishes["poor"] | dirty["poor"] | type["poor"], time["lower"] | temperature["average"])
# seklinde yapmak istedim fakat hataya cozum bulamadigim icin ayri ayri gerceklestirdim
# ornegi yeterince kavradigimi dusunerek sadece time ve temperature degerleri icin yaptim
# tesekkurler

# %%
