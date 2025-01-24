from setuptools import setup, find_packages

setup(
    name="kotaro", # nombre único de paquete
    version="0.1.0-dev", # primera versión
    author="César Medina", # Mi nombre
    author_email="cameding@gmail.com",
    description="A simple package for data cleaning and visualization",
    packages=find_packages(),  # Busca automáticamente las carpetas con módulos
    python_requires=">=3.11.7",
    install_requires=[
        "numpy==2.2.2",
        "pandas==2.2.3",
        "pytest==7.4.0",
        "scikit_learn==1.2.2",
        "setuptools==68.2.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/tu_usuario/kotaro",  # Link a tu repositorio
    keywords=["data cleaning", "visualization", "data analysis"],
    license="MIT",
)