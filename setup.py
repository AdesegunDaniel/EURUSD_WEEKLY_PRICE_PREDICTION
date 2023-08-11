from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''this return a list of all the requirements that we will be using in this model'''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.read().splitlines()
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements


setup(name="EURUSD PRICE PREDICTIONS",
      version="0.0.1",
      author="ADESESEGUN OLUWADEMILADE",
      author_email="adesegundemilade11@gmail.com",
      packages= find_packages(),
      install_requires=get_requirements('requirements.txt')
      )