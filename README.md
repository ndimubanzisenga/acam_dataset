This is a dataset of microphone array signals and camera images collected using an Acoustic Camera (ACAM) for two building acoustics scenarios: the open-air scenario and the building scenario.
The microphone array used is a 40cm*40cm array, with 40 microphones. The setup arrangement for each scenario is illustrated by figures in the "doc" folder. This folder also contains the report of the project for which this dataset was created, where more details of the experimental setup for the creation of this dataset is given.
Each dataset consists of 5 files:
* acam_array_40.xml : Microphone array layout
* CcmConfiguration.xml : ACAM configuration used for data collection
* MicRawSignals.txt : Signals of the 40 microphones in the array
* OpticalImages.avi : Scene Images recorded by the ACAM optical camera
* PinkNoise_10s_50kHz.txt : Original sound signal played on the speaker

This dataset can be used to test sound source localization techniques based on microphone arrays. The images of the scene provide the ground truth against which the accuracy of the tested sound source localization techniques can be compared.
A Python script to test this dataset is provided in the "src" folder. This requires the installation of the "Acoular" framework. To run this script the dataset name and the sound source localization method to be used should be passed as arguments.
`python Bftest.py leakage 1`
In this test implementation:
* 1 : runs the test with the Funtional Beamforming method
* 2 : runs the test with the Conventional Beamforming method
* 3 : runs the test with the Clean-SC method
