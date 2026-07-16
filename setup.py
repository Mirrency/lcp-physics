from setuptools import find_packages, setup

setup(
    name='lcp_physics',
    version='0.1.0',
    description='A differentiable LCP physics engine in PyTorch.',
    author='Filipe de Avila Belbute-Peres',
    author_email='filiped@cs.cmu.edu',
    platforms=['any'],
    url='https://github.com/locuslab/lcp-physics',
    packages=find_packages(exclude=['demos', 'videos']),
    python_requires='>=3.11,<3.12',
    install_requires=[
        'torch==2.7.1',
        'numpy==1.26.4',
        'pygame==2.6.1',
    ],
    extras_require={'test': ['pytest==8.4.1']},
)
