from setuptools import setup, find_packages


setup(name='VisDA-ECCV20',
      version='0.1.0',
      description='Solution for VisDA Challenge in ECCV 2020',
      author='Yixiao Ge',
      author_email='geyixiao831@gmail.com',
      url='https://github.com/yxgeee/VisDA-ECCV20',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification'
      ])
