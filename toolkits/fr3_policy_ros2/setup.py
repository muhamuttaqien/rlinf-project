from setuptools import setup

package_name = "rlinf_fr3_policy_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="RLinf Authors",
    maintainer_email="contact@rlinf.ai",
    description="Non-actuating FR3 policy ROS2 shadow integration (M5 only)",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "fr3_policy_shadow = rlinf_fr3_policy_bridge.node:main",
        "fr3_policy_shadow_smoke = rlinf_fr3_policy_bridge.smoke:main",
    ]},
)
