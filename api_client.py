#!/usr/bin/python3
# -*- coding: utf-8 -*-
import openai
import time
import logging
import codecs

logger = logging.getLogger(__name__)

# This code is partially adapted from [Ye et al. (2023)](https://arxiv.org/pdf/2302.05698.pdf)
# Original code: [https://github.com/HKUNLP/icl-ceil](https://github.com/HKUNLP/icl-ceil
# Modifications: [Brief description of your modifications]


class OpenAIClient:
    def __init__(self, **kwargs):
        self.provider = kwargs.pop('provider')
        self.model = kwargs.pop('model')
        if self.provider == 'azure':
            url = "https://{}.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
            self.api_key = kwargs["api_key"]
            self.api_version = "2023-05-15"
            self.api_type = "azure"
            self.deployment_name = kwargs['deployment_name']
            self.azure_openai_endpoint = url.format(kwargs['resource'])
            self.n_processes = 2
        else:
            self.api_key = kwargs["api_key"]
            self.n_processes = 1

    def call_api(self, prompt: str, max_tokens=200, temperature=1,
                 stop=None, n=None, echo=False):
        result = None
        if temperature == 0:
            n = 1
        if stop:
            stop = stop.copy()
            for i, s in enumerate(stop):
                if '\\' in s:
                    # hydra reads \n to \\n, here we decode it back to \n
                    stop[i] = codecs.decode(s, 'unicode_escape')
            stop = list(stop)
        while result is None:
            try:
                if self.provider == 'azure':
                    openai.api_key = self.api_key
                    openai.api_base = self.azure_openai_endpoint
                    openai.api_type = self.api_type
                    openai.api_version = self.api_version
                    result = openai.Completion.create(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n,
                        stop=stop,
                        #logprobs=1,
                        echo=echo,
                        engine=self.deployment_name
                    )
                else:
                    result = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        api_key=self.api_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        n=n,
                        stop=stop,
                        #logprobs=1,
                        echo=echo
                    )
                time.sleep(1)
                return result
            except Exception as e:
                #if e.http_status == 400:
                #    logger.info(f"Content filter, return empty response.")
                #    return None
                logger.info(f"{str(e)}, 'Retry.")
                time.sleep(4)

    def extract_response(self, response):
        texts = [r['text'] for r in response['choices']]
        return [{"text": text} for text in zip(texts)]


def run_api(prompt, **kwargs):
    client = kwargs.pop('client')
    kwargs.update({"echo": False, "max_tokens": 10})
    response = client.call_api(prompt=prompt, **kwargs)
    #response = client.extract_response(response)
    return response
