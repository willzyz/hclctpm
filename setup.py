from setuptools import setup, find_packages

version = '1.0.0'
def requirements(filename="requirements.txt"):
    with open(filename) as f:
        return f.readlines()
setup(
    name='hscls',
    version=version,
    description='HSCLS: Heterogeneous Sequential Casual Learning System.',
    author='Michelangelo Team',
    author_email='michelangelo@uber.com',
    url='gitolite@code.uber.internal:data/deeplearning_hscls',
    packages=find_packages(exclude=['tests', 'tests.*']),
    install_requires=requirements(),
    zip_safe=False
    #classifiers=['Private :: Do Not Upload']
)
