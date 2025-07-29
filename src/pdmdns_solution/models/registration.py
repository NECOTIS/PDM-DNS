class ModelBank:
    models = {}

    @staticmethod
    def register(model_name, model_class, args_callback=None):
        ModelBank.models[model_name] = (model_class, args_callback)

    @staticmethod
    def get_models():
        return ModelBank.models

    @staticmethod
    def setup_args(subparser) -> None:
        """Set-up model-specific args for CLI"""
        for model_name, (_, callback) in ModelBank.models.items():
            new_parser = subparser.add_parser(model_name)
            if callback is not None:
                callback(new_parser)

    @staticmethod
    def get_model_class(model_name):
        return ModelBank.models[model_name][0]
