from pathlib import Path
from setuptools import find_packages, setup

setup(
    name='dex-general-motion-retargeting', version='0.1.0',
    description='Latest base Walker Tienkung Dex, 29 DOF with a fixed head',
    long_description=Path('README.md').read_text(), long_description_content_type='text/markdown',
    packages=find_packages(), python_requires='>=3.11',
    install_requires=['mujoco==3.10.0', 'mink==1.3.0', 'qpsolvers[daqp]>=4.5',
                      'numpy>=1.26', 'scipy>=1.14', 'rich', 'tqdm', 'loop-rate-limiters',
                      'imageio[ffmpeg]', 'smplx==0.1.28', 'torch>=2.7'],
)
