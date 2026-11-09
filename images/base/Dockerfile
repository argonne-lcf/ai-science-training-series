FROM jupyterhub/singleuser:5.4.2

USER root

RUN apt-get update && apt-get install -y emacs-nox

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --root-user-action=ignore -r /tmp/requirements.txt

USER $NB_UID