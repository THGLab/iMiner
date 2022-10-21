'''
Author: Jie Li
Date Created: Oct 21, 2022

A consensus docking module for iMiner
'''

from project import BaseProject

class ConsensusDocking(BaseProject):
    def __init__(self, project_name, project_path=None, docking_protocols=None) -> None:
        '''
        Initialize a project with a project name and a project path

        When project_path is None, the project will be initialized in the current working directory,
        using the project_name as the project folder name

        :param docking_protocols: list of docking protocols to be used for consensus docking, list of ["vina", "vina-gpu", "ad4", "icm"]
        '''
        super().__init__(project_name, project_path)