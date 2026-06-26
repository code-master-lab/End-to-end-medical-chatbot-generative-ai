from pathlib import Path

from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).parent


def read_readme():
    readme_path = ROOT_DIR / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "End-to-end medical chatbot using generative AI."


def get_requirements(filename):
    requirements_path = ROOT_DIR / filename
    if not requirements_path.exists():
        return []

    requirements = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


setup(
    name="generative-ai-medical-chatbot",
    version="0.1.0",
    author="Vivek Raut",
    author_email="rautlata59@gmail.com",
    description=(
        "End-to-end medical chatbot using FastAPI, LangChain, Pinecone, "
        "Groq, and HuggingFace embeddings"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Framework :: FastAPI",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: ASGI :: Application",
    ],
)
