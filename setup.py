from setuptools import setup, find_packages

version = '0.0.1'

setup(
    name='ReactionAutoencoder',
    version=version,
    packages=find_packages(),
    url='https://github.com/TimurGimadiev/ReactionAutoencoder',
    license='LGPLv3',
    author=['Dr. Timur Gimadiev'],
    author_email=['timur.gimadiev@gmail.com', 'nougmanoff@protonmail.com'],
    python_requires='>=3.8.0',
    install_requires=['tqdm>=4.61.0', 'torch>=1.9', 'numpy', 'scikit-learn'],
    classifiers=['Environment :: Plugins',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.9',
                 ]
)

