FROM quay.io/jupyter/base-notebook:latest

USER root

RUN apt-get update

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --root-user-action=ignore -r /tmp/requirements.txt

USER $NB_UID