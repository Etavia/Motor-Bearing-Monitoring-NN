# Motor-Bearing-Monitoring-NN
Electric motor bearing monitoring based on sound utilizing neural networks.

This code demonstrates detection of a faulty bearing in an electric motor by the sound it makes. Detecting faults in industrial environment is important because failure in some critical component could cause the entire plant to halt. Production losses of downtime can be easily millions of euros. It is possible to detect faults by measuring the vibration of the motors shaft or acceleration. Advantage of sound based detection is cheap cost of microphones. They do not require direct contact with the motor and are therefore more easier to install. However, loud background noise typical in industrial plants can cause problems to sound detection.

Sound of the motors was recorded earlier and sound processing was continued with using MATLAB. Different frequencies of sound was then extracted from the signal using Fast Fourier Transform (FFT). By plotting frequency domain faulty signal can be easily recognized by its 3300 Hz fault frequency and higher amplitude. Each signal was then divided into 100 samples. Each sample was further downsampled to a vector of lenght of 20 rows. Downsampling was made by getting highest value from sampling window. Purpose of downsampling was to make machine learning algorithm more efficient by reducing the number of inputs.

![frequency plot](https://user-images.githubusercontent.com/55585889/123843032-1931c000-d91a-11eb-96f7-75dc724c0ce7.png)

Data of faulty and normal signals is then used to train feedforward neural network. Pattern of signals is so clear that network reaches nearly 100 % accuracy in less than 10 epochs.

Summary of code flow
1. Load sound data
2. Split sound data to 100 samples stored in a matrix
3. Convert samples from time domain to frequency domain by using FFT
4. Downsample signal to 20 samples by getting highest value from sampling window
5. Plot signal from each step to demonstrate what has been done so far
6. Train neural network
7. Print result of performance test

Advantages of machine learning
- Can be trained to detect different kind of fault categories like bearing, stator winding, load and rotor bar
- Can be trained to analyze different signals than sound like vibration for instance
- Method can be applied to many other applications than electric motors 

Video about laboratory setting of normal and faulty motor. Detecting obvious fault by listening is easy. See if you can hear which motor is healthy and which one is faulty.
https://www.youtube.com/watch?v=YBFgkpnamAI

Repository contains MATLAB code and pre-recorded 5 second samples of faulty motor and normal motor in MATLAB audio format.

Prerequisites for using this code
- Matlab
- Deep learning toolbox
