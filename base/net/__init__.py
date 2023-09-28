from .models import Backbone


def get_model(model_name, num_label):
    return Backbone(num_label, model_name)
