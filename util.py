openai_models = ["text-davinci-003", "gpt-3.5", "gpt-4"]

seq2seq_models = ['allenai/tk-instruct-3b-def-pos', 'allenai/tk-instruct-3b-def', 'allenai/tk-instruct-base-def-pos',
                  'google/flan-t5-base', 'google/flan-t5-small', 'google/flan-t5-large', 'google/flan-t5-xl',
                  'google/t5-efficient-mini', 't5-large', 't5-3b', 'allenai/tk-instruct-small-def-pos',
                  'allenai/tk-instruct-large-def-pos']
# > 6B
large_seq2seq_models = ['allenai/tk-instruct-11b-def', 'allenai/tk-instruct-11b-def-pos', 'google/flan-t5-xxl',
                        'google/flan-ul2', 't5-11b', 'bigscience/T0pp']

causal_models = ['databricks/dolly-v2-3b', 'facebook/opt-iml-1.3b']

large_causal_models = ["EleutherAI/gpt-neox-20b", '/storage/ukp/shared/shared_model_weights/models--alpaca/7B',
'/storage/ukp/shared/shared_model_weights/models--llama/7B','/storage/ukp/shared/shared_model_weights/models--llama/13B',
'/storage/ukp/shared/shared_model_weights/models--llama/30B', '/storage/ukp/shared/shared_model_weights/models--llama/65B',
                       'facebook/opt-iml-30b', 'databricks/dolly-v2-7b', 'databricks/dolly-v2-12b']


sociodemographic_attribute_values = {
    'gender': ['male', 'female'],  # for now, we ignore 'nonbinary'
    'race':['Black or African American', 'White', 'Asian',
       'American Indian or Alaska Native', 'Hispanic',
       'Native Hawaiian or Pacific Islander'],
    'education': ['Some college but no degree',
       'Associate degree in college (2-year)',
       "Bachelor's degree in college (4-year)", 'Doctoral degree',
       "Master's degree", 'Professional degree (JD, MD)',
       'High school graduate (high school diploma or equivalent including GED)',
       'Less than high school degree'],
    'age': ['35 - 44', '18 - 24', '25 - 34', '45 - 54', '55 - 64', '65 or older', 'Under 18'],
    'political_affiliation': ['Liberal', 'Independent', 'Conservative']
}

choices_per_dataset = {
    'toxicity_diverse_perspectives': ['not toxic', 'slightly toxic', 'moderately toxic', 'very toxic', 'extremely toxic'],
    'sentiment_diaz': ['Very positive', 'Somewhat positive', 'Neutral', 'Somewhat negative', 'Very negative'],
    'toxicity_jigsaw': ['yes', 'no'],
    'stance_semeval2016t6': ['favor', 'against', 'none'],
    'stance_gwsd': ['agree', 'disagree', 'neutral'],
    'hatespeech_twitter': ['sexism', 'racism', 'both', 'neither'],
    'hatespeech_ghc': ['yes', 'no']
}

label_maps = {'toxicity_diverse_perspectives': ['not toxic', 'slightly toxic', 'moderately toxic', 'very toxic', 'extremely toxic'],
             'sentiment_diaz': ['very negative', 'somewhat negative', 'neutral', 'somewhat positive', 'very positive'],
             'toxicity_jigsaw': ['0', '1'],
              'stance_semeval2016t6': ['favor', 'against', 'none'],
              'stance_gwsd': ['agree', 'disagree', 'neutral'],
              'hatespeech_twitter': ['sexism', 'racism', 'both', 'neither'],
              'hatespeech_ghc': ['0', '1']
             }

binary_mapping = {
    '0': 'no',
    '1': 'yes',
    0: 'no',
    1: 'yes',
}

fewshot_templates = {
    'toxicity_diverse_perspectives': {
        'with_sd': """A {sd_info} rated the degree of toxicity in the following text as '{{label}}':\nText: '{{text}}'\n""",
        'without_sd': """A person rated the degree of toxicity in the following text as '{label}':\nText: '{text}'\n""",
    },
    'sentiment_diaz': {
        'with_sd': """A {sd_info} classified the sentiment of the following text as '{{label}}':\nText: '{{text}}'\n""",
        'without_sd': """A person classified the sentiment of the following text as '{label}':\nText: '{text}'\n""",
    },
    'toxicity_jigsaw': {
        'with_sd': """A {sd_info} classified if the following text is toxic as '{{label}}':\nText: '{{text}}'\n""",
        'without_sd': """A person classified if the following text is toxic as '{label}':\nText: '{text}'\n""",
    },
    'stance_semeval2016t6': {
        'with_sd': """A {sd_info} classified the stance of the following statement about {{topic}} as '{{label}}':\nStatement: '{{text}}'\n""",
        'without_sd': """A person classified the stance of the following statement about {topic} as '{label}':\nStatement: '{text}'\n""",
    },
    'stance_gwsd': {
        'with_sd': """A {sd_info} classified the stance of the following statement about {{topic}} as '{{label}}':\nStatement: '{{text}}'\n""",
        'without_sd': """A person classified the stance of the following statement about {topic} as '{label}':\nStatement: '{text}'\n""",
    },
    'hatespeech_twitter': {
        'with_sd': """A {sd_info} classified the type of hate speech in the following text as '{{label}}':\nText: '{{text}}'\n""",
        'without_sd': """A person classified the type of hate speech in the following text as '{label}':\nText: '{text}'\n""",
    },
    'hatespeech_ghc': {
        'with_sd': """A {sd_info} classified if the following text contains hatespeech as '{{label}}':\nText: '{{text}}'\n""",
        'without_sd': """A person classified if the following text contains hatespeech as '{label}':\nText: '{text}'\n""",
    }
}

dataset2prompt = {
    'toxicity_diverse_perspectives': {
        'with_sd': ["""Given a text, how would a {sd_info} rate the degree of toxicity in the text. Possible values are 'not toxic', 'slightly toxic', 'moderately toxic', 'very toxic' or 'extremely toxic'.\nText: '{{text}}'\nToxicity:""",
                    """Imagine you are a {sd_info}. Given a text, how would you rate the degree of toxicity in the text. Possible values are 'not toxic', 'slightly toxic', 'moderately toxic', 'very toxic' or 'extremely toxic'.\nText: '{{text}}'\nToxicity:""",
                    """Person: {sd_info}\nText: '{{text}}'\nToxicity ('not toxic'|'slightly toxic'|'moderately toxic'|'very toxic'|'extremely toxic'):"""
                    ],
        'without_sd': ["""Given a text, how would a person rate the degree of toxicity in the text. Possible values are 'not toxic', 'slightly toxic', 'moderately toxic', 'very toxic' or 'extremely toxic'.\nText: '{text}'\nToxicity:""",
                       """Imagine you are a person who is asked to rate the degree of toxicity in the text. Possible values are 'not toxic', 'slightly toxic', 'moderately toxic', 'very toxic' or 'extremely toxic'.\nText: '{text}'\nToxicity:""",
                       """Text: '{text}'\nToxicity ('not toxic'|'slightly toxic'|'moderately toxic'|'very toxic'|'extremely toxic'):"""
                       ]
    },
    'sentiment_diaz': {
        'with_sd': ["""Given a text, how would a {sd_info} classify the sentiment of the text. Possible values are 'Very positive', 'Somewhat positive', 'Neutral', 'Somewhat negative' or 'Very negative'.\nText: '{{text}}'\nSentiment:""",
                    """Imagine you are a {sd_info}. Given a text, how would you classify the sentiment of the text. Possible values are 'Very positive', 'Somewhat positive', 'Neutral', 'Somewhat negative' or 'Very negative'.\nText: '{{text}}'\nSentiment:""",
                    """Person: {sd_info}\nText: '{{text}}'\nSentiment ('Very positive'|'Somewhat positive'|'Neutral'|'Somewhat negative'|'Very negative'):"""
                    ],
        'without_sd': ["""Given a text, how would a person classify the sentiment of the text. Possible values are 'Very positive', 'Somewhat positive', 'Neutral', 'Somewhat negative' or 'Very negative'.\nText: '{text}'\nSentiment:""",
                          """Imagine you are a person who is asked to classify the sentiment of the text. Possible values are 'Very positive', 'Somewhat positive', 'Neutral', 'Somewhat negative' or 'Very negative'.\nText: '{text}'\nSentiment:""",
                       """Text: '{text}'\nSentiment ('Very positive'|'Somewhat positive'|'Neutral'|'Somewhat negative'|'Very negative'):"""],
    },
    'toxicity_jigsaw': {
        'with_sd': [
            """Given a text, how would a {sd_info} classify if the text is toxic. Possible values are 'yes' or 'no'.\nText: '{{text}}'\nToxic:""",
            """Imagine you are a {sd_info}. Given a text, how would you classify if the text is toxic. Possible values are 'yes' or 'no'.\nText: '{{text}}'\nToxic:""",
            """Person: {sd_info}\nText: '{{text}}'\nToxic ('yes'|'no'):"""],
        'without_sd': [
            """Given a text, how would a person classify if the text is toxic. Possible values are 'yes' or 'no'.\nText: '{text}'\nToxic:""",
            """Imagine you are a person who is asked to classify if the text is toxic. Possible values are 'yes' or 'no'.\nText: '{text}'\nToxic:""",
            """Text: '{text}'\nToxic ('yes'|'no'):"""],
    },
    'stance_semeval2016t6': {
        'with_sd': [
            """Given a statement about {{topic}}, how would a {sd_info} classify the stance of the statement. Possible values are 'favor', 'against' or 'none'.\nStatement: '{{text}}'\nStance:""",
            """Imagine you are a {sd_info}. Given a statement about {{topic}}, how would you classify the stance of the statement. Possible values are 'favor', 'against' or 'none'.\nStatement: '{{text}}'\nStance:""",
            """Person: {sd_info}\nStatement about {{topic}}: '{{text}}'\nStance ('favor'|'against'|'none'):"""],
        'without_sd': [
            """Given a statement about {topic}, how would a person classify the stance of the statement. Possible values are 'favor', 'against' or 'none'.\nStatement: '{text}'\nStance:""",
            """Imagine you are a person who is asked to classify the stance of a statement about {topic}. Possible values are 'favor', 'against' or 'none'.\nStatement: '{text}'\nStance:""",
            """Statement about {topic}: '{text}'\nStance ('favor'|'against'|'none'):"""],
    },
    'stance_gwsd': {
        'with_sd': [
            """Given a statement about the topic '{{topic}}', how would a {sd_info} classify the stance of the statement. Possible values are 'agree', 'disagree' or 'neutral'.\nStatement: '{{text}}'\nStance:""",
            """Imagine you are a {sd_info}. Given a statement about the topic '{{topic}}', how would you classify the stance of the statement. Possible values are 'agree', 'disagree' or 'neutral'.\nStatement: '{{text}}'\nStance:""",
            """Person: {sd_info}\nStatement about the topic '{{topic}}': '{{text}}'\nStance ('agree'|'disagree'|'neutral'):"""],
        'without_sd': [
            """Given a statement about the topic '{topic}', how would a person classify the stance of the statement. Possible values are 'agree', 'disagree' or 'neutral'.\nStatement: '{text}'\nStance:""",
            """Imagine you are a person who is asked to classify the stance of a statement about the topic '{topic}'. Possible values are 'agree', 'disagree' or 'neutral'.\nStatement: '{text}'\nStance:""",
            """Statement about the topic '{topic}': '{text}'\nStance ('agree'|'disagree'|'neutral'):"""],
    },
    'hatespeech_twitter': {
        'with_sd': [
            """Given a text, how would a {sd_info} classify the type of hatespeech the text contains. Possible values are 'sexism', 'racism', 'both' or 'neither'.\nText: '{{text}}'\nHatespeech:""",
            """Imagine you are a {sd_info}. Given a text, how would you classify the type of hatespeech the text contains. Possible values are 'sexism', 'racism', 'both' or 'neither'.\nText: '{{text}}'\nHatespeech:""",
            """Person: {sd_info}\nText: '{{text}}'\nHatespeech ('sexism'|'racism'|'both'|'neither'):"""],
        'without_sd': [
            """Given a text, how would a person classify the type of hatespeech the text contains. Possible values are 'sexism', 'racism', 'both' or 'neither'.\nText: '{text}'\nHatespeech:""",
            """Imagine you are a person who is asked to classify the type of hatespeech the text contains. Possible values are 'sexism', 'racism', 'both' or 'neither'.\nText: '{text}'\nHatespeech:""",
            """Text: '{text}'\nHatespeech ('sexism'|'racism'|'both'|'neither'):"""],
    },
    'hatespeech_ghc': {
        'with_sd': [
            """Given a text, how would a {sd_info} classify if the text contains hatespeech. Possible values are 'yes' or 'no'.\nText: '{{text}}'\nHatespeech:""",
            """Imagine you are a {sd_info}. Given a text, how would you classify if the text contains hatespeech. Possible values are 'yes' or 'no'.\nText: '{{text}}'\nHatespeech:""",
            """Person: {sd_info}\nText: '{{text}}'\nHatespeech ('yes'|'no'):"""],
        'without_sd': [
            """Given a text, how would a person classify if the text contains hatespeech. Possible values are 'yes' or 'no'.\nText: '{text}'\nHatespeech:""",
            """Imagine you are a person who is asked to classify if the text contains hatespeech. Possible values are 'yes' or 'no'.\nText: '{text}'\nHatespeech:""",
            """Text: '{text}'\nHatespeech ('yes'|'no'):"""],
    }
}


def compose_sociodemographic_info(sociodemographic_attributes):
    def check_last_item(x, nr_items):
        if nr_items == 1:
            x += """ and"""
        if nr_items > 1:
            x += ""","""
        return x
    # order of attributes: gender, race, age, education, political_affiliation
    sd_info = """person of"""
    assert len(sociodemographic_attributes) > 0
    nr_items = len(sociodemographic_attributes)
    if set(sociodemographic_attribute_values.keys()) == set(sociodemographic_attributes) or sociodemographic_attributes == ['ratings']:
        return """person of gender '{gender}', race '{race}', age '{age}', education level '{education}' and political affiliation '{political_affiliation}'"""
    for att in sociodemographic_attribute_values.keys():
        if att in sociodemographic_attributes:
            sd_info += """ {att} '{{{att}}}'""".format(att=att)
            nr_items -= 1
            sd_info = check_last_item(sd_info, nr_items)
    return sd_info


def format_prompt(cfg, prompt, row, text_column):
    dataset = cfg.task_name
    sociodemographic_info = cfg.sociodemographics
    if sociodemographic_info:
        if 'age_range' in row and 'age' not in row:
            row['age'] = row['age_range']  # prompt template is expecting 'age'
        if 'political_affilation' in row and 'political_affiliation' not in row:
            row['political_affiliation'] = row['political_affilation']  # prompt template is expecting 'political_affiliation'
        prompt_attributes = {}
        for att in sociodemographic_attribute_values.keys():  # only consider the attributes which are relevant
            if att in row:
                prompt_attributes[att] = row[att]
        if dataset in ['stance_semeval2016t6', 'stance_gwsd']:
            return prompt.format(topic=row['target'], **prompt_attributes, text=row[text_column])
        else:
            return prompt.format(**prompt_attributes, text=row[text_column])
    else:
        if dataset in ['stance_semeval2016t6', 'stance_gwsd']:
            return prompt.format(topic=row['target'], text=row[text_column])
        else:
            return prompt.format(text=row[text_column])