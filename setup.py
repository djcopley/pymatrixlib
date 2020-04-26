from setuptools import setup

setup(
    author="Daniel Copley",
    author_email="djrcopley@gmail.com",

    name="pymatrixlib",
    license="GPLv3",
    description="A python-based matrix library.",
    keywords=["linear algebra", "matrix", "calculator", "library", "python3"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Topic :: Application",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    python_requires=">=3=4",
    setup_requires=[
        "setuptools_scm"
    ],
    install_requires=[
        "numpy",
    ],
    packages=[
        "pymatrixlib",
    ],
    use_scm_version={
        "relative_to": __file__,
        "write_to": "pymatrixlib/version.py"
    }
)
