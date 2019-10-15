from setuptools import setup

setup(
	name='segnet',
	version='0.0.0',
	description='Keras implementation of SegNet',
	url='https://github.com/ImmortalEmperor/SegNet',
	author='Aidan Possemiers',
	author_email='apossem@gmail.com',
	license='MIT',
	packages=['segnet'],
	zip_safe=True,
	install_requires = ['keras']
)
