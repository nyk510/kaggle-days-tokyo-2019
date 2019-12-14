FROM registry.gitlab.com/nyker510/analysis-template/cpu
# FROM registry.gitlab.com/nyker510/analysis-template/gpu:45e6cc9a5c4fb3a469b74bb197005c8536ab44b6

ENV PATH=${PATH}:/home/penguin/.local/bin
USER root
ADD requirements.txt requirements.txt
RUN pip uninstall anemone vivid && pip install -U pip && pip install -r requirements.txt

USER penguin
