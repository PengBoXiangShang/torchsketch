from setuptools import find_packages, setup

def _get_long_description():
    with open('README.md', 'r') as f:
        README = f.read()
    return README



setup(
    name = "torchsketch",
    version = "0.1.0",
    license = 'MIT',
    author = "Peng Xu",
    author_email = "peng.xu@bupt.edu.cn",
    description = "TorchSketch is an open source software library for free-hand sketch oriented deep learning research, which is built on the top of PyTorch.",
    long_description = _get_long_description(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/PengBoXiangShang/torchsketch/",
    keywords=['sketch', 'free-hand sketch', 'pytorch', 'deep learning'],
    packages = find_packages(exclude=['docs']),
    install_requires = [
        'torchvision==0.4.0',
        'torchnet==0.0.4',
        'torch==1.2.0',
        'Pillow<7.0.0',
        'gdown',
        'pdf2image',
        'pyfiglet',
        'pyunpack',
        'scipy',
        'wget',
        'CairoSVG',
        'numpy',
        'imageio',
        'patool'
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
    ],
)