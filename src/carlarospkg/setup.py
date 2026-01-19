from setuptools import setup

package_name = 'carlarospkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='krg6',
    maintainer_email='krg6@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'carlarosnode = carlarospkg.carlarosnode:main',
            'camshakenode = carlarospkg.camshakenode:main',
            'carlarosnode_log1 = carlarospkg.carlarosnode_log1:main',
            'carlarosnode_copy = carlarospkg.carlarosnode_copy:main',
            'pilotnetsim_log = carlarospkg.pilotnetsim_log:main',
            'carlarosnode_Sync_Log = carlarospkg.carlarosnode_Sync_log:main',
            'carlarosnode_ASync_Log = carlarospkg.carlarosnode_ASync_log:main'
        ],
    },
)
