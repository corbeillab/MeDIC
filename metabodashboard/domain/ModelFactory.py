import importlib
import os
import sklearn

from .MetaboModel import MetaboModel
from ..conf.SupportedModels import LEARN_CONFIG

#TODO: deals with methods names' that are used in Results (for example), how to retrieve features/importance/etc
class ModelFactory:
    def __init__(self):
        pass

    def create_supported_models(self) -> dict:
        supported_models = {}
        for model_name, model_configuration in LEARN_CONFIG.items():
            supported_models[model_name] = MetaboModel(model_configuration["function"], model_configuration["ParamGrid"])
        return supported_models

    def _get_model_from_import(self, imports_list: list, model_name: str) -> sklearn:
        last_import = importlib.import_module("." + imports_list[0], package="sklearn")

        for next_import in imports_list[1:]:
            last_import = getattr(last_import, next_import)

        model = getattr(last_import, model_name)
        return model

    def create_custom_model(self, model_name: str, needed_imports: str, grid_search_param: dict) -> MetaboModel:
        imports_list = needed_imports.split(".")
        model = self._get_model_from_import(imports_list, model_name)
        return MetaboModel(model, grid_search_param)
