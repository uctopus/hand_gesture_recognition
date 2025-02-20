from setuptools import setup, find_packages

setup(
    name="hand_gesture_recognition",
    version="0.1.0",
    description="Radar-based Hand Gesture Recognition using Deep Learning",
    author="Filip Chodorowski",
    author_email="filipchodorowski@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.10",
) 