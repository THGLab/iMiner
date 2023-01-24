import setuptools
from os import path
import iMiner

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='iMiner',
        version=iMiner.__version__,
        author='Jie Li',
        author_email='jerry-li1996@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/THGLab/iMiner',
        },
        description=
        "Inhibitor Mining protocol (iMiner) including Consensus Docking and MD Analysis",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            'Drug Discovery', 'Docking', 'Quantum Chemistry',
            'Molecular Dynamics'
        ],
        license='MIT',
        packages=setuptools.find_packages(exclue=["tests"]),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
    )