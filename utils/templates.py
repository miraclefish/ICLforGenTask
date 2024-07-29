from openicl import PromptTemplate


class TemplateBank:

    def __init__(self):

        pass

    def get_template(self, dataset_name):

        if dataset_name in ['SST-5', 'Amazon', 'Yelp']:
            return self.sentiment_classification_5()
        elif dataset_name in ['SST-2', 'MR', 'CR']:
            return self.sentiment_classification_2()

    
    def sentiment_classification_2(self):

        template = PromptTemplate(template={
                                                    0: '</E></text>\nIt was terrible .\n\n',
                                                    1: '</E></text>\nIt was great .\n\n'
                                                },
                                    column_token_map={'sentence' : '</text>'},
                                    ice_token='</E>'
                    )
        return template

    def sentiment_classification_5(self):

        template = PromptTemplate(template={
                                                    0: '</E></text>\nIt was terrible .\n\n',
                                                    1: '</E></text>\nIt was bad .\n\n',
                                                    2: '</E></text>\nIt was okay .\n\n',
                                                    3: '</E></text>\nIt was good .\n\n',
                                                    4: '</E></text>\nIt was great .\n\n'
                                                },
                                    column_token_map={'sentence' : '</text>'},
                                    ice_token='</E>'
                    )
        return template