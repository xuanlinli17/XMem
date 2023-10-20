from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='XMem',
        version='0.0.1',
        description="XMem mask tracking",
        install_requires=[],
        packages=find_packages(include=["XMem*"]),
    )
