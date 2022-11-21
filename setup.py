from setuptools import setup, find_packages

setup(
   name='modelo_susceptibilidad',
   version='0.1',
   description='Modulo del modelo de susceptibilidad a incendios de cobertura vegetal en el valle de Aburrá de SIATA',
   author='Sebastian Ospina, Esneider Zapata, Paola Montoya',
   author_email='seospina@gmail.com, paomon5@gmail.com',
   packages=find_packages(),  #find_packages(include=['mypythonlib']) ## esto es para buscar el paquete que quiero instlar, lo puedo dejar co parentesisi vacíos
   include_package_data=True,
   install_requires=['numpy', 'pandas', 'matplotlib', 'cartopy', 'netCDF4', "xarray", "cfgrib"], #external packages as dependencies
)