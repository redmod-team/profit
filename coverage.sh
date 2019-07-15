#!/bin/sh

python3-coverage run setup.py test && python3-coverage xml && CODACY_PROJECT_TOKEN=308aa0ce5b304661b46a848dc7b946a3 python-codacy-coverage

