import setuptools
import datetime
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()


def setup(scm=None):
    setuptools.setup(
        name='iMiner',
        use_scm_version=scm,
        setup_requires=["setuptools_scm"],
        python_requires=">=3.8",
        author='Jie Li',
        author_email='jerry-li1996@berkeley.edu',
        project_urls={
            'Source': 'https://github.com/THGLab/iMiner',
        },
        description="Inhibitor Mining protocol (iMiner) including Consensus Docking and MD Analysis",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=[
            'Drug Discovery', 'Docking', 'Quantum Chemistry',
            'Molecular Dynamics'
        ],
        license='MIT',
        packages=setuptools.find_packages(exclude=["tests"]),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.8',
        ],
        zip_safe=False,
        entry_points={
            "console_scripts": [
                "iMiner = iMiner.entrypoints.main:main"
            ]
        }
    )


today = datetime.date.today().strftime("%b-%d-%Y")
with open("iMiner/_date.py", 'w') as fp:
    fp.write(f'date = "{today}"')

try:
    setup(scm={"write_to": f"iMiner/_version.py"})
except:
    setup(scm=None)