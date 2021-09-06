import setuptools

setuptools.setup(
    name='locomotionbench',
    version='0.1.0',
    author='Felix Aller',
    author_email='felix.aller@ziti.uni-heidelberg.de',
    description='Compute metrics based on gait segmentation for periodic walking motions and export PI',
    url='https://gitlab.com/orb-benchmarking/eb_hum_bench/',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6.0',
    scripts=['script/run_i3sa.py'],
    install_requires=[
        "wheel",
        "csaps",
        "cycler",
        "Cython",
        "hdbscan",
        "joblib",
        "kiwisolver",
        "llvmlite",
        "matplotlib",
        "numpy",
        "pandas",
        "Pillow",
        "pyparsing",
        "python-dateutil",
        "pytz",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "Shapely",
        "six",
        "sklearn",
        "threadpoolctl"
    ]
)
