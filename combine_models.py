import pandas as pd

# combine the results of all clustering models


class combine_models:
    def __init__(self, official_class, model_class_list):
        assert len(model_class_list) > 0
        combine_class = official_class.merge(
            model_class_list[0].output_clusters_df(), on=['company'], how='right')
        for i in range(1, len(model_class_list)):
            combine_class = combine_class.merge(
                model_class_list[i].output_clusters_df(), on=['company'])
        self.combine_class = combine_class[~combine_class['official_class'].isna(
        )]
        self.namelist = ['official_class'] + [i.name for i in model_class_list]
