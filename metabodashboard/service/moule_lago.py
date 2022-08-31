class lago:
    def __init__(self, methodfit, methodpredict, method):
        pass

    def fit(self, algo):
        return algo.lago


class my_class:
    def method1(self):
        return "Hello World"

    def method2(self, methodToRun):
        result = methodToRun()
        return result


obj = my_class()
# method1 is passed as an argument
print(obj.method2(obj.method1))

# TODO : a faire -> on demande d'aller rentrer les params dans fichier de conf, avec les noms des méthodes et des attributs nécessaires
# puis on fourni ça à la classe "moule"
